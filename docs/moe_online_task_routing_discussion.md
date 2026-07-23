# 基于在线任务 Embedding 的 PPO-MoE 专家路由

> 讨论整理与研究设计草案，2026-07-22  
> 范围：SP Tracking 当前 SPV5-1 residual MoE、基于梯度的任务发现、Humanoid-GPT HME、局部 reference 聚类及在线分组。  
> 状态：本文区分了仓库中已经存在的实现、论文明确提出的方法和仍需实验验证的设计假设；推荐方案尚未实现。
>
> **2026-07-23 后续调研更新**：GMT 与 HoloMotion 的新证据使首版方案收敛为“reference-only 直接 Top-2 路由”，而不是默认加入 state correction 或 SwAV/prototypes。完整论证、仓库改造点和实验矩阵见 [`moe_ref_window_online_routing_research.md`](./moe_ref_window_online_routing_research.md)。本文保留为原始讨论记录。

## 1. 问题与目标

当前问题不是给已有动作类别训练一个分类器，而是在包含成千上万条未知动作的数据中自动发现适合不同专家学习的局部控制模式。这里需要特别避免把“任务”误解为 motion 文件或人工语义标签。

本文采用以下工作定义：

- **motion**：一条完整参考动作序列；
- **局部任务或控制模式**：某个有限时间窗口内具有相似参考动力学、接触结构和控制需求的样本集合；
- **任务 embedding**：由局部窗口得到的连续向量；
- **cluster / expert ID**：任务 embedding 在离散原型空间中的分组结果；
- **router**：把当前样本映射到一个或多个 experts 的模块。

核心约束是：一条 motion 内部可能发生任务切换。因此最终分类单位应是时间窗口或 rollout segment，而不是完整 motion。一个跳跃序列可以依次经历助跑、起跳、腾空、落地和恢复，每一段可以路由到不同 expert。

希望达到的结果是：

1. 不依赖人工动作类别或 motion ID；
2. 能在 motion 内在线切换分组；
3. 8 个 experts 获得稳定、互补的专长，而不是随机分工或路由坍缩；
4. 路由既反映参考运动模式，也能处理当前机器人偏离参考轨迹后的恢复需求；
5. PPO 的优化目标、expert 专长和任务 embedding 最终具有可测量的一致性。

## 2. 当前仓库中的 MoE 与 reference 管线

本节是对当前代码的事实性描述，主要对应：

- [`residual_moe.py`](../src/sp_tracking/tasks/tracking/rl/residual_moe.py)
- [`spv5_1_models.py`](../src/sp_tracking/tasks/tracking/rl/spv5_1_models.py)
- [`spv5_models.py`](../src/sp_tracking/tasks/tracking/rl/spv5_models.py)
- [`spv5.py`](../src/sp_tracking/tasks/tracking/mdp/spv5.py)
- [`ppo.py`](../src/sp_tracking/tasks/tracking/rl/ppo.py)
- [`tracking_bfm_spv5_1_moe_actor_heft_critic_heft_reward.yaml`](../src/sp_tracking/conf/task/tracking_bfm_spv5_1_moe_actor_heft_critic_heft_reward.yaml)

### 2.1 当前 residual MoE

当前 actor MoE 是 observation-conditioned residual MoE：

```text
完整且归一化的 SPV5-1 policy feature（1651 维）
    ↓
context encoder：1651 → 1472 → 608
    ↓
shared residual block
    ├── linear router：608 → 8
    │       ↓ softmax，temperature = 1.5
    │       ↓ Top-2 并重新归一化
    └── 8 个 residual experts
            ↓ Top-2 加权求和
shared feature + mixed expert residual
    ↓ RMSNorm + action output head
```

当前实现具有以下性质：

- router 输入是完整 policy feature，包含机器人本体状态、参考状态、tracking error、key-body 状态和估计接触等信息，并不是纯 reference router；
- expert 数量为 8，Top-K 为 2；
- 为保持导出简单，前向时会计算全部 8 个 expert，但只有 Top-2 residual 得到非零混合权重；
- shared backbone 对所有样本更新，被选中的 experts 对对应样本更新；
- router 初始权重是小随机量，避免全零 logits 导致所有样本初始选择相同 experts；
- 当前目标主要让 router 从 PPO 回报中自行形成路由，没有显式“任务类别”监督。

### 2.2 当前 router 正则化

`SPV51ContactEstimatorMoEPPO` 已实现两类 router 正则：

- 对完整 rollout 上的平均 dense routing probability 计算相对均匀分布的 KL，用于负载均衡；
- 可选的低熵置信度损失，用 warm-up 和 ramp 控制。

当前 actor MoE 配置中：

```text
moe_balance_loss_coef = 0.01
moe_confidence_loss_coef = 0.0
```

因此当前训练会约束全局使用率，但不会直接要求单个样本形成高置信度任务分配，也不会保证同一个 expert 学到语义或控制上连贯的模式。负载均衡只能防止明显的使用率坍缩，不能自动定义“什么是任务”。

### 2.3 当前 SPV5 reference 信息

actor 可见的 noisy reference window 为：

```text
reference steps = [-42, -41, ..., 0, ..., +7]
窗口长度 = 50 帧
每帧 = root position 3 + root rotation 6D + joint position 29 = 38 维
```

现有 reference encoder 从这 50 帧 noisy qpos 重建 `[-3, +7]` 共 11 帧 clean support，并由 `SPV5ReferenceKinematics` 推导速度、重力、key-body 等 policy feature。最终标准 reference policy feature 主要使用 `[0, +4]` 共 5 帧，当前 key-body 速度计算使用局部 support。

这意味着：

- 当前输入已经具有足够长的局部时间上下文，可以训练 reference task encoder；
- 不应只使用 policy 最终的 5 帧 reference feature 提取周期/节奏，原始 50 帧窗口更合适；
- 当前 rollout-only clean target 只暴露 11 帧。如果要用完整 clean 50 帧生成聚类 teacher，需要新增 rollout-only clean window，或者在离线 motion 数据上生成 teacher；
- 当前 reference encoder 受显式去噪监督，PPO policy gradient 不会穿过它。第一版任务 encoder 最好与现有去噪 encoder 分开，避免在线聚类干扰已有监督目标。

## 3. 基于 PPO 梯度分类任务的原始设想

### 3.1 核心直觉

原始设想是：如果两个局部样本对策略参数提出相近的优化方向，那么它们适合交给同一个 expert；如果梯度方向持续冲突，则应分配给不同 experts。因此可以对 PPO 梯度做在线聚类，得到 8 个任务组，并让每组梯度只更新对应 expert 和共享 backbone。

这个直觉有合理之处。人工动作语义不一定等价于控制优化关系，而梯度直接描述“当前样本希望怎样改变策略”。相关多任务学习研究也会用一个任务的梯度对另一个任务损失的影响来估计 task affinity [12]。

### 3.2 “窗口聚合梯度”是什么

窗口聚合不是给完整 motion 做一个永久签名，而是在时刻 `t` 附近选择一个局部集合 `W_t`，对其中样本的梯度做降噪聚合。概念上可以写成：

```text
单样本梯度：gᵢ = Project(∇θ_probe L_actor,i)

窗口 embedding：eₜ = Normalize(Σᵢ∈Wₜ wᵢ gᵢ / Σᵢ∈Wₜ wᵢ)
```

其中：

- `L_actor,i` 可以采用 advantage normalization 之后的实际 PPO actor objective；
- `θ_probe` 是用于比较任务的固定参数子集或共享中间表示；
- `Project` 是随机投影、分层统计或低秩压缩，避免保存百万维完整梯度；
- `W_t` 可以是同一轨迹的短时间窗口，也可以是 reference embedding 相近的一组 rollout 样本；
- `w_i` 可按有效样本、优势幅度或时间核加权，但必须防止梯度大小完全支配方向。

这个 `e_t` 随时间变化，因此允许一条 motion 在不同任务之间切换。

### 3.3 对谁求梯度

不应直接拼接各 expert 参数的梯度作为通用任务签名，因为不同样本经过的 expert 不同，参数坐标不可直接比较，而且结果会被现有路由强烈影响。更合理的候选是：

1. **共享表示梯度**：对 shared feature `h_t` 求 `∂L_actor/∂h_t`；
2. **共享 probe 层参数梯度**：对固定的小型共享投影层或 action output head 求梯度；
3. **共享 backbone 的压缩梯度**：只选若干层并随机投影；
4. **expert competence 统计**：比较各 expert 在相同窗口上的 PPO surrogate、log-probability 或 value error，作为比原始梯度更直接的“哪个 expert 更适合”信号。

即使动作由不同 experts 产生，也仍然可以对共享层求梯度。被选 expert 的输出依赖 shared feature，而 shared feature 又依赖 shared backbone，因此 PPO loss 可以沿着：

```text
PPO loss
  → action distribution
  → selected expert residual
  → shared feature
  → shared backbone parameters
```

反向传播。不过，“能够求到”不等于“适合作为独立任务标签”。selected expert 和 router 已经改变了反传到 shared feature 的梯度，所以该梯度仍带有当前专家分工的影响。

### 3.4 梯度作为主聚类信号的根本风险

如果从训练第一步就用实际 MoE PPO 梯度决定 cluster，会产生循环依赖：

```text
当前 router 选择 expert
    → expert 决定动作与 PPO loss
    → expert 决定梯度形状
    → 梯度聚类决定下一次 router
    → 早期随机差异被持续放大
```

主要风险包括：

- **循环定义**：任务标签依赖尚未训练好的 expert，expert 又依赖任务标签；
- **高方差**：PPO gradient 同时受 advantage、采样动作、clip、critic 误差和 episode phase 影响；
- **非平稳**：策略参数变化后，同一 reference window 的梯度签名会漂移；
- **尺度混淆**：困难样本或大 tracking error 可能仅因梯度范数更大而形成一类；
- **初始锁定**：早期随机路由可能形成自证式专家专长；
- **计算与存储成本**：逐样本完整参数梯度不可接受。

因此当前更可靠的判断是：

> 梯度适合用于检验“分组是否具有优化意义”，也可以在专家形成初步专长后作为二级修正信号；不适合从随机初始化开始充当唯一的任务定义。

仓库已有 [`policy_gradient_diagnostics.md`](policy_gradient_diagnostics.md)，能够在两个已知 motion 条件下测量 actor/critic 梯度余弦、范数和 cancellation。它可以作为后续验证工具的基础，但目前依赖已知 simple/hard 标签，还不是无监督在线分组器。

## 4. Humanoid-GPT 如何分类 motion

Humanoid-GPT [1] 的分类方法是 Harmonic Motion Embedding（HME），建立在 DeepPhase 的 Periodic Autoencoder [2] 思路上。其流程是：

```text
完整 retarget motion sequence
    ↓
在不同数据分区上训练的 Periodic Autoencoders
    ↓
提取各关节周期幅值和频率
    ↓
对整条序列的关节谐波特征聚合均值与标准差
    ↓
一个 motion-level HME 向量
    ↓
K-Means
    ↓
一个完整 motion 对应一个 cluster
```

论文正文报告大约 300 个 clusters，每个 cluster 约有 1000–2000 条 motion sequences。附录比较了 `K ∈ {128, 256, 384, 512, 1024}`，约 384 个 experts 在簇内一致性、覆盖范围和计算成本之间表现最好。聚类太粗时，一个 expert 需要覆盖过于异质的运动；过细时，训练成本和后续蒸馏冲突增加。

聚类后，论文为每个 cluster 单独训练一个 PPO motion expert，最后通过 DAgger 将 expert library 蒸馏成一个 causal Transformer generalist。最终部署的 Humanoid-GPT 并不是“clusterer + online MoE router”。

### 4.1 HME 能支持什么

- 自动按运动节奏和关节周期结构组织大规模 motion 数据；
- 不需要人工动作类别；
- 为每个 PPO expert 提供相对一致的离线训练子集；
- 幅值、频率相对不敏感于动作当前处于周期的哪个 phase。

### 4.2 HME 不能直接解决什么

- 一个 motion 只有一个 cluster ID，不能在 motion 内切换；
- HME 是离线 sequence-level 聚类，不是在线 task inference；
- 周期幅值和频率对走、跑、舞蹈等周期动作很有效，但不能保证充分描述起跳、落地、摔倒恢复等非周期瞬态；
- 论文的成功还包含数百个独立 experts、大规模数据、DAgger 蒸馏和 Transformer generalist，不能直接证明“8 个在线 residual experts + HME router”一定有效。

### 4.3 适合当前任务的局部 HME 改造

将完整 motion HME 改为滑动窗口 HME：

```text
Rₜ = reference[t-42 : t+7]
    ↓
去除全局平移和绝对朝向
    ↓
局部 temporal encoder / Periodic Autoencoder
    ↓
local embedding zₜ
    ↓
8 个 prototypes
    ↓
随时间变化的 cluster distribution qₜ
```

局部输入除了周期特征，还应加入或让网络推断：

- root-frame 下的相对位移；
- root height、rotation、linear/angular velocity；
- joint position、velocity，必要时加入 acceleration；
- key-body position/velocity；
- 脚部速度或 reference contact proxy；
- 多时间尺度的变化特征。

需要去除全局 XY 和绝对 heading，防止聚类结果变成“处于哪个位置、朝哪个方向”，而不是运动模式。

## 5. 在线 embedding、边训练边分组的方法

“在线 embedding”有两种含义：一是 embedding 与 clusters 在流式训练数据上同步更新；二是在 episode 内根据新观测持续推断当前潜在任务。已有研究分别覆盖这两种问题。

| 路线 | 代表方法 | 在线信号 | 能否处理任务切换 | expert 数量 | 对当前问题的适用性 |
| --- | --- | --- | --- | --- | --- |
| 在线原型聚类 | SwAV [3]、ODC [4] | 当前 minibatch 的 embedding | 可以，取决于窗口定义 | 固定 | 最适合固定 8 experts 的第一版 |
| 离散码本 | VQ-VAE [5] | encoder 与 codebook 联合学习 | 可以 | 固定 | 实现直接，但需防 codebook collapse |
| 概率任务 latent | PEARL [6]、VariBAD [7] | 最近的 `(s,a,r,s')` context | 可以在线适应 | 通常固定 | 更像控制情境推断，不是天然离散分组 |
| 组合任务推断 | OCEAN [8] | global/local context | 明确建模子任务转移 | 通常固定 | 与 motion 内多阶段切换最接近 |
| 在线变点检测 | MOCA [9] | 时序预测似然与 task run length | 擅长检测离散切换 | 不直接绑定 experts | 可作为路由切换先验，但假设较强 |
| 非参数混合模型 | Nagabandi 等 [10]、CN-DPM [11] | 现有组件解释能力 | 可以创建新组件 | 动态增长 | 适合未知任务数，工程复杂度较高 |
| 梯度 task affinity | TAG [12] | 一个任务梯度对另一个任务损失的影响 | 原方法不做匿名逐窗口推断 | 预定义任务 | 适合启发梯度验证，不是直接路由器 |

### 5.1 在线原型聚类

SwAV 同时学习表示和 prototypes，并要求同一个样本的不同增强视图预测一致的 cluster assignment；训练时用平衡分配抑制坍缩。ODC 则维护动态样本标签和中心记忆，通过 minibatch 级重分配让网络与 cluster centers 同步演化。

映射到本项目：

```text
noisy local reference Rₜ
    ↓ online encoder Eθ
normalized embedding zₜ
    ↓ 与 8 个 prototypes 比较
balanced pseudo-label qₜ
    ↓
Top-2 routing + cluster consistency training
```

训练时可以在大规模并行环境的 batch 上做平衡分配；单机器人部署时不能依赖跨样本 Sinkhorn，因此需要使用 prototype softmax/nearest prototype，或者训练一个小型 student router 模仿训练期 balanced assignment。

### 5.2 在线概率任务推断

PEARL 和 VariBAD 从最近经验推断连续任务 belief；OCEAN 同时建模全局任务和局部子任务 latent，局部 latent 可以在 episode 中变化，因此概念上最接近“一条 motion 内连续出现不同控制阶段”。但这些方法主要使用实际交互 context，而不是只使用 reference。

对当前系统而言，实际交互 context 可以提供 reference 不包含的信息，例如：

- 机器人当前 tracking error；
- 是否受推、打滑或失衡；
- 当前接触与动力学状态；
- 某个 expert 在当前状态下是否已经失效。

因此它更适合作为 reference 路由的状态修正器，而不是完全替代 reference task encoder。

### 5.3 动态创建 experts

Chinese Restaurant Process 或 Dirichlet Process mixture 可以在已有 experts 都无法解释新数据时创建组件，并在旧任务再次出现时召回旧组件。这与“任务数量未知”最一致，但当前阶段不建议优先采用，因为连续 motion 的平滑变化容易被误判为大量新任务，而且需要解决 expert 创建、初始化、合并、淘汰和分布式同步。

第一阶段固定 `K = 8`，并比较 `K = 4/8/16`，比直接引入动态 expert 数量更容易判断核心假设是否成立。

## 6. 推荐方案：reference 先验 + 状态修正 + 梯度验证

推荐结构不是纯 reference hard router，也不是纯 PPO gradient clusterer，而是三层信息分工：

```text
局部 reference window
    ↓ ReferenceTaskEncoder + online prototypes
参考任务先验 p_ref(k | Rₜ)
                    ┐
机器人状态、tracking error、contact
    ↓ StateCorrectionRouter
状态修正 Δₜ(k)     ┘
    ↓
final logits = log(p_ref + ε) + Δₜ
    ↓ temporal prior / hysteresis
    ↓ Top-2
selected experts
```

分工是：

- reference encoder 回答“当前想执行哪种局部运动”；
- state correction 回答“机器人当前是否需要偏离典型执行方式来恢复”；
- 梯度和 expert competence 回答“当前分组是否真的减少优化冲突、形成了控制专长”。

### 6.1 ReferenceTaskEncoder

建议第一版新增独立 task encoder，不直接替换现有去噪 reference encoder：

```text
输入：50 × 38 noisy reference window
主干：Temporal Conv 或小型 causal/bidirectional Transformer
输出：建议从 64 维 normalized embedding 开始
```

这里的 64 维只是工程起点，不是论文结论。输入增强可以包括：

- reference noise 的两个独立采样；
- 小范围时间裁剪或偏移；
- 合理的 time warp；
- 左右镜像，但必须同步处理关节映射；
- 全局平移和 heading 变化。

若部署时 reference 的未来 `+1...+7` 已知，可以继续使用当前非因果窗口；若未来部署只提供当前和历史 reference，则训练和部署都必须改为 causal 输入，避免信息契约不一致。

### 6.2 Online/target encoder 与 prototypes

为了减小表示漂移，采用 online encoder 和 EMA target encoder：

```text
z_online = E_online(view_1(Rₜ))
z_target = stop_grad(E_target(view_2(Rₜ)))

E_target ← m × E_target + (1-m) × E_online
```

维护 8 个 normalized prototypes `c₁...c₈`。训练 batch 中根据 `z_target · c_k / τ` 生成 balanced assignment；online encoder 预测该 assignment。prototypes 也应慢速更新，防止 cluster identity 在相邻 PPO updates 中频繁置换。

如果能够增加完整 clean reference teacher，可用 clean window 产生 target assignment、让 noisy window 预测它；如果暂时不改 observation contract，则可以从 noisy window 的两种增强视图做一致性训练。

### 6.3 Top-2 而不是 hard Top-1

在任务边界附近使用 Top-2 有三个优势：

- 两个相邻模式可以平滑共享控制；
- cluster center 尚未稳定时不必做不可逆的单 expert 选择；
- 与当前 residual MoE 实现一致，改造范围较小。

为了避免逐帧抖动，可在 logits 上加入上一时刻分布的弱先验，或者对 assignment 使用短时 EMA。平滑强度不能过大，否则会延迟真正的起跳、接触或恢复切换。更完整的版本可以借鉴 OCEAN 的 local latent 或 MOCA 的变点概率，仅在检测到明显模式改变时降低历史先验。

### 6.4 在线 EM 式训练

整体过程可以理解为交替进行的在线 EM：

**E-step：在线分组**

1. 用本轮冻结的 target encoder/prototypes 计算 `q_t`；
2. 加入负载平衡和弱时间先验；
3. 生成 Top-2 assignment；
4. 将 assignment 或生成它所需的 target 信息随 rollout 固定保存。

**M-step：更新表示与策略**

1. 用固定 assignment 完成本轮 PPO epochs；
2. 更新被选 experts、shared backbone 和 state correction router；
3. 更新 task encoder 的自监督/聚类目标；
4. PPO update 结束后慢速更新 target encoder 和 prototypes。

固定每个 rollout 的聚类 target 很重要。如果同一个 rollout 样本在多个 PPO epoch 内不断重新分组，expert 身份会成为额外的非平稳因素，PPO ratio 和专家训练数据都会剧烈变化。

### 6.5 推荐损失

可以从以下组合开始：

```text
L_total = L_PPO
        + λ_pred × L_reference_prediction
        + λ_cluster × L_swapped_cluster_consistency
        + λ_balance × L_load_balance
        + λ_temporal × L_temporal_consistency
        + λ_comp × L_expert_competence
```

各项含义：

- `L_PPO`：策略和价值学习主目标；
- `L_reference_prediction`：重建 clean support 或预测后续 reference，使 embedding 保留局部动力学；
- `L_swapped_cluster_consistency`：同一 reference 的不同增强视图应预测相同 cluster；
- `L_load_balance`：维持 8 个 prototypes/experts 的有效使用率；
- `L_temporal_consistency`：抑制无意义的逐帧路由抖动；
- `L_expert_competence`：在 experts 形成初步差异后，让更适合当前窗口的 expert 获得更高概率。

`L_expert_competence` 不应从训练第一步启用。可在 warm-up 后使用按窗口聚合并 detach 的 expert-specific PPO surrogate、log-probability、critic error 或梯度 affinity，再通过 EMA 降噪。

### 6.6 双时间尺度与稳定性

“边训练边分组”不意味着所有模块必须同速更新。建议至少使用：

```text
快速：policy shared backbone、selected experts
中速：online task encoder、student router
慢速：target encoder、cluster prototypes、competence statistics
```

否则容易出现：embedding 改变 → cluster 改变 → expert 数据改变 → PPO gradient 改变 → embedding 再改变的高速正反馈。

短暂 representation warm-up 并不违背在线学习：数据仍按流式方式到达，只是在 task embedding 尚未形成结构之前避免 hard specialization。warm-up 期间可以使用均衡随机 Top-2、soft routing 或当前 observation router。

## 7. 梯度在推荐方案中的位置

梯度不再承担初始任务定义，而承担三项更可靠的职责。

### 7.1 验证 cluster 是否具有控制意义

对 reference cluster 得到的样本，比较：

- cluster 内 actor gradient cosine；
- cluster 间 actor gradient cosine；
- cluster 内/间 gradient cancellation；
- 梯度范数是否仅由 tracking difficulty 决定；
- 不同 episode phase、接触状态和 return 下结果是否稳定。

理想结果不是“所有 cluster 内梯度完全同向”，而是 cluster 内亲和度显著高于随机分组和 cluster 间亲和度。

### 7.2 精炼 prototypes

在 reference embedding 已稳定后，可加入弱约束：梯度兼容的窗口在 embedding 空间更近，持续冲突的窗口更远。该约束必须使用 target/EMA gradient statistics，并限制权重，避免 PPO 高方差覆盖 reference 表示。

### 7.3 检验专家专长

对同一窗口进行 counterfactual expert 评估：比较各 expert 的 surrogate、action likelihood、短期 return 或 value error。如果 expert `k` 只在被分配数据上表现好，而互换 expert 明显变差，才能说明专长真实存在；仅观察路由使用率不足以证明任务发现成功。

## 8. 实验路线

### 阶段 0：建立基线

至少保留以下基线：

1. 当前完整 observation router；
2. 随机但负载均衡的 Top-2 router；
3. 纯 reference MLP router，不做显式聚类；
4. 离线 local-HME/K-Means router；
5. 在线 prototype router；
6. 在线 prototype + state correction；
7. 在线 prototype + state correction + gradient/competence refinement。

同时比较 `K = 4/8/16`。Humanoid-GPT 的消融说明聚类粒度很重要，但它使用的是完整独立 experts 和数百 clusters，不能据此直接确定本项目的最优 `K`。

### 阶段 1：只验证 embedding 与在线分组

在不改变 PPO 路由前，记录：

- cluster 使用率与 effective experts；
- assignment entropy；
- 同一窗口不同噪声视图的一致性；
- cluster switch rate 与平均持续时间；
- cluster 与 root/joint velocity、contact proxy、motion phase 的关系；
- 新 motion 上到最近 prototype 的距离；
- prototype identity 是否随训练大幅漂移。

这一阶段不要求 cluster 对应“走、跑、跳”等人工名称。需要证明的是分组稳定、非坍缩、对噪声鲁棒，并且保留局部动力学差异。

### 阶段 2：冻结分组训练 experts

先冻结 target encoder/prototypes，用固定在线规则训练 experts，观察：

- held-out motion tracking return/success；
- 各 cluster 的 per-expert 表现矩阵；
- route boundary 附近的动作连续性；
- 当前 MoE baseline 与 reference router 的差异；
- 是否出现某些 experts 只学习困难度而非运动结构。

### 阶段 3：联合微调

在初步专长形成后逐步启用：

1. state correction；
2. temporal change prior；
3. expert competence；
4. 弱 gradient-affinity refinement。

每增加一项都需要单独消融，不能只比较最终完整系统和当前 baseline，否则无法判断收益来自哪里。

### 8.1 主要评估指标

**控制性能**

- tracking reward、success/fall rate；
- held-out motion 和高动态 motion；
- 扰动、打滑、噪声下恢复；
- per-motion 和 per-phase 尾部性能，而不只看总体平均。

**路由质量**

- expert usage、effective experts、Top-1 probability；
- switch rate、边界延迟、短周期抖动；
- noise/time-crop assignment consistency；
- prototype 漂移和空 cluster 数量。

**专长与优化关系**

- cluster 内/间梯度余弦和 cancellation；
- expert × cluster counterfactual 表现矩阵；
- 同 cluster 共享 expert 是否优于随机共享；
- state correction 是否只在 tracking error 大时显著改变 reference prior。

所有关键实验应使用多个 paired seeds，并同时报告计算量。在线分组增加 encoder、prototype 更新和诊断成本，性能提升必须与额外预算区分。

## 9. 主要风险与缓解措施

| 风险 | 现象 | 缓解措施 |
| --- | --- | --- |
| Cluster collapse | 大部分样本进入少数 experts | balanced assignment、全 rollout balance KL、空簇重置 |
| Prototype 身份漂移 | expert 语义在训练中不断交换 | EMA target、慢速中心更新、prototype matching、固定 rollout assignment |
| Rich-get-richer | 早期更强 expert 获得更多数据并继续变强 | warm-up、均衡配额、competence 延迟启用、探索性 Top-2 |
| 路由抖动 | 相邻帧频繁切换 experts | temporal prior、短时 EMA、边界感知的 switch loss |
| 切换过迟 | temporal smoothing 把真实切换抹平 | 变点分数降低历史先验、限制 smoothing 强度 |
| 按困难度而非任务聚类 | 大误差/大梯度样本单独成簇 | reference 为主、embedding 归一化、分层检查 tracking error |
| 纯 reference 无法恢复 | 相同 reference 在正常与失衡状态下需不同控制 | state correction router、tracking error/contact 输入 |
| PPO 与聚类共同非平稳 | loss、cluster、expert 同时追逐 | 在线 EM、双时间尺度、每轮冻结 target/assignment |
| 周期表示漏掉瞬态 | 跳跃/恢复被错误合并 | temporal encoder + 预测目标，不只使用 amplitude/frequency |
| 训练/部署输入不一致 | 训练使用未来或 clean reference，部署不可用 | 明确 observation contract，用 noisy student 模仿 teacher |
| K 选择错误 | 8 类过粗或过细 | `K=4/8/16` 消融，必要时再研究动态 mixtures |

## 10. 当前建议与暂不建议

### 建议优先实施

1. 保持固定 8 experts 和 Top-2，新增独立 `ReferenceTaskEncoder`；
2. 使用当前 50 帧 reference window，先做坐标规范化；
3. 用 SwAV/ODC 风格的 balanced online prototypes；
4. 采用 EMA target encoder，按 rollout 固定 assignment；
5. 先训练 reference-only 路由基线，再加入 state correction；
6. 用现有 gradient diagnostics 扩展验证 cluster 内/间优化亲和度；
7. 完整比较当前 observation router、随机路由、离线 local-HME 和在线 prototypes。

### 暂不建议

1. 从随机初始化开始用实际 MoE PPO 梯度作为唯一 cluster embedding；
2. 直接对完整 motion 分配一个永久 expert ID；
3. 只使用单帧 reference state；
4. 只靠负载均衡期待 experts 自动获得连贯语义；
5. 第一版就动态创建和删除 experts；
6. 在每个 PPO minibatch/epoch 内重新改变同一 rollout 样本的 hard assignment；
7. 直接把 Humanoid-GPT 的 sequence-level HME 结果视为在线 router 的实验证据。

## 11. 尚待确认的工程选择

以下选择不会改变总体路线，但会影响具体实现：

1. 部署端是否始终能看到 `+1...+7` 的未来 reference；
2. 是否愿意新增 rollout-only clean 50 帧 teacher observation；
3. task encoder 使用独立 Temporal Conv、Transformer，还是暴露现有 reference encoder 的 penultimate feature；
4. cluster assignment 是否作为 rollout metadata 显式存储；
5. state correction router 的输入范围和最大修正幅度；
6. competence 使用 actor surrogate、critic error、短期 return 还是梯度 affinity；
7. actor MoE 和 critic MoE 是否共享 task prototypes，还是第一阶段只改 actor。

## 12. 参考文献

1. Zekun Qi et al. **Humanoid-GPT: Scaling Data and Structure for Zero-Shot Motion Tracking**. arXiv, 2026. [arXiv](https://arxiv.org/abs/2606.03985)；[本地 PDF](</home/lenovo/Zotero/storage/4I5WP736/Qi 等 - 2026 - Humanoid-GPT Scaling Data and Structure for Zero-Shot Motion Tracking.pdf>)。本文主要参考其 HME、motion cluster、PPO expert library 和 DAgger distillation 设计。
2. Sebastian Starke, Ian Mason, Taku Komura. **DeepPhase: Periodic Autoencoders for Learning Motion Phase Manifolds**. ACM Transactions on Graphics, 2022. [DOI](https://doi.org/10.1145/3528223.3530178)；[作者 PDF](https://i.cs.hku.hk/~taku/deepphase.pdf)。Humanoid-GPT 的 Periodic Autoencoder/HME 思路来源之一。
3. Mathilde Caron et al. **Unsupervised Learning of Visual Features by Contrasting Cluster Assignments**. NeurIPS, 2020. [论文页面](https://proceedings.neurips.cc/paper/2020/hash/70feb62b69f16e0238f741fab228fec2-Abstract.html)。SwAV 同时学习表示和在线 cluster assignments，并使用平衡分配防止坍缩。
4. Xiaohang Zhan et al. **Online Deep Clustering for Unsupervised Representation Learning**. ECCV, 2020. [arXiv](https://arxiv.org/abs/2006.10645)。ODC 通过动态样本/中心记忆做 minibatch 级在线重分配。
5. Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu. **Neural Discrete Representation Learning**. NeurIPS, 2017. [arXiv](https://arxiv.org/abs/1711.00937)。VQ-VAE 提供可端到端学习的离散 codebook 表示。
6. Kate Rakelly et al. **Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables**. ICML, 2019. [PMLR](https://proceedings.mlr.press/v97/rakelly19a.html)。PEARL 从交互 context 在线推断概率任务变量。
7. Luisa Zintgraf et al. **VariBAD: A Very Good Method for Bayes-Adaptive Deep RL via Meta-Learning**. ICLR, 2020. [arXiv](https://arxiv.org/abs/1910.08348)。使用变分 belief 表示未知任务和不确定性。
8. Hongyu Ren et al. **OCEAN: Online Task Inference for Compositional Tasks with Context Adaptation**. UAI, 2020. [PMLR](https://proceedings.mlr.press/v124/ren20a.html)。global/local latent 的设计直接面向多阶段组合任务和子任务切换。
9. James Harrison et al. **Continuous Meta-Learning without Tasks**. NeurIPS, 2020. [论文页面](https://proceedings.neurips.cc/paper/2020/hash/cc3f5463bc4d26bc38eadc8bcffbc654-Abstract.html)。MOCA 把可微贝叶斯变点检测与 meta-learning 结合，用于无任务边界的时间序列。
10. Anusha Nagabandi, Chelsea Finn, Sergey Levine. **Deep Online Learning Via Meta-Learning: Continual Adaptation for Model-Based RL**. ICLR, 2019. [OpenReview](https://openreview.net/forum?id=HyxAfnA5tm)。使用在线 EM 和 Chinese Restaurant Process mixture 处理非平稳任务并动态维护模型组件。
11. Soochan Lee et al. **A Neural Dirichlet Process Mixture Model for Task-Free Continual Learning**. ICLR, 2020. [OpenReview PDF](https://openreview.net/pdf?id=SJxSOJStPr)。CN-DPM 在不知道 task ID 和边界时动态扩展 neural experts。
12. Chris Fifty et al. **Efficiently Identifying Task Groupings for Multi-Task Learning**. NeurIPS, 2021. [论文页面](https://proceedings.neurips.cc/paper_files/paper/2021/hash/e77910ebb93b511588557806310f78f1-Abstract.html)。通过一个任务的梯度对另一个任务损失的影响估计 task affinity；适合作为梯度分组思想的参考，但不是匿名在线 RL router。

## 13. 一句话结论

> 对当前 SP Tracking，最值得先验证的路线是：用 50 帧局部 reference 在线学习平衡的离散任务 prototypes，以它作为 Top-2 expert 路由先验；再用机器人状态修正路由，并把 PPO 梯度用于验证和后期精炼，而不是用尚未稳定的专家梯度从训练第一步定义任务。
