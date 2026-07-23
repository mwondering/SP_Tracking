# SP Tracking：训练中在线分类 ref motion window 的文献与实现调研

> 调研日期：2026-07-23  
> 关联讨论：[`moe_online_task_routing_discussion.md`](./moe_online_task_routing_discussion.md)  
> 范围：当前 SPV5-1 residual MoE、局部 reference window 路由、PPO 联合训练、路由稳定性与专家专长验证。  
> 状态：研究结论与实现设计，尚未实现。

## 1. 结论

“边训练边给 ref motion window 分类”是当前最值得优先验证的方向，但这里的“分类”应作如下精确定义：

- 分类单位是当前时刻的局部 reference window，不是整条 motion；
- 类别是 8 个专家上的可学习 Top-2 分配，不要求预先具有“走、跑、跳”等人工语义；
- reference 决定“应该调用哪类控制专长”，机器人状态决定“所选专长此刻应输出什么动作”；
- 第一版直接用 PPO 端到端学习 reference router，不先引入 SwAV、在线 EM 或动态增删专家；
- state correction 不应作为首版默认设计，应在 reference-only 基线失败后作为受限消融。

这一判断的证据强度并不相同：

1. **直接支持**：GMT 的门控会在同一复合 motion 内随局部阶段切换，说明 MoE 路由不必绑定整条 motion。
2. **最强工程支持**：HoloMotion 报告，使用 proprioception 与 reference 共同路由会因低层动力学扰动产生路由振荡和实机退化，因此把 router 限制为只看 reference；完整机器人状态仍进入策略主干。其公开配置也采用 reference-only Top-2，而没有启用显式聚类标签、负载均衡或相邻帧切换惩罚。
3. **间接支持**：Humanoid-GPT 使用运动表示对完整序列离线聚类，再训练约 300 个 PPO 专家，证明运动结构可以形成有控制意义的专家划分；但它不是在线窗口路由。
4. **仍待验证**：50 帧是否优于更短窗口、8 类是否合适、显式 prototypes 是否优于直接 gate，以及 reference-only 是否会损害强扰动恢复，都不能由现有论文替本仓库回答。

因此，推荐的首个可证伪假设是：

> 在专家和共享主干仍接收完整 policy feature 的前提下，只用规范化后的 50 帧 noisy reference window 产生 Top-2 权重，相比当前 full-observation router，能够降低由机器人状态扰动引起的路由抖动，并形成更稳定的局部运动专长，同时不降低 tracking 与恢复性能。

## 2. 文献与官方实现证据

### 2.1 与当前问题最接近的证据

| 工作 | 已确认做法 | 对当前方向的支持 | 不能推出的结论 |
| --- | --- | --- | --- |
| [GMT](https://arxiv.org/html/2506.14770) | 门控与专家同时接收机器人状态和 motion target；复合动作可视化中，门控权重随站立、踢腿、后退等阶段切换 | 支持“同一 motion 内按局部阶段路由”，反对整条 motion 固定一个 expert ID | 不支持 reference-only；没有显式在线聚类标签 |
| [HoloMotion-1](https://arxiv.org/html/2605.15336) | 稀疏 MoE Transformer；论文明确报告 full observation routing 在实机上会因低层动力学变化而振荡，改为只用 reference 路由 | 直接支持“reference 决定专家身份，状态留在策略主干” | 其规模、Transformer 结构和数据量与本仓库不同，不能直接照搬超参数 |
| [Humanoid-GPT](https://arxiv.org/html/2606.03985) | 用 Periodic Autoencoder 提取 HME，对整段序列做 K-Means，约 300 个 cluster 各自训练 PPO expert，再蒸馏到单一 Transformer | 支持 motion embedding、专家粒度和数据均衡具有控制意义 | 是离线 sequence-level 分类，不是训练中的 window router |
| [DeepPhase](https://doi.org/10.1145/3528223.3530178) | 无监督学习多维周期相位流形，特征距离可改善运动的时空对齐与相似性度量 | 支持从运动时间结构学习表示 | 周期归纳偏置对落地、失衡恢复等瞬态模式不充分 |

GMT 是“窗口内切换”的直接现象证据，HoloMotion 是“reference-only 路由”的直接工程证据。二者结合后，当前讨论中“reference prior + 无约束 state correction”不再是首选；更稳妥的默认结构是：

```text
reference window ───────────────→ expert identity / Top-2 weights

完整 policy feature ─→ shared backbone ─→ selected expert residuals ─→ action
```

HoloMotion 当前公开实现还提供了三个重要细节：

- [motion tracking MoE 配置](https://github.com/HorizonRobotics/HoloMotion/blob/71ab7e976de23aa9bb351030dd61b05426d3443c/holomotion/config/modules/motion_tracking/motion_tracking_tf-moe.yaml)明确列出 router 只接收 reference/command 类字段，并排除 robot state 与 tracking error；
- 配置为 1024 个 routed experts、1 个 shared expert、Top-2；这与本仓库“共享主干 + 8 个 residual experts”的结构思想兼容，但稀疏规模完全不同；
- [PPO 配置](https://github.com/HorizonRobotics/HoloMotion/blob/71ab7e976de23aa9bb351030dd61b05426d3443c/holomotion/config/algo/ppo_tf.yaml)启用了状态/运动学辅助预测和 inactive-expert margin，但关闭了 uniform load balance、router 重建、相邻帧切换惩罚和 dynamic bias。

最后一点尤其重要：公开系统不是先产生一个独立的运动学 cluster label 再训练 gate，而是让 reference-conditioned gate 与策略共同学习；显式“分类”是 Top-2 expert assignment 本身。

### 2.2 路由稳定性与时间序列表示

[StableMoE](https://aclanthology.org/2022.acl-long.489/)指出，同一输入在训练过程中不断改变目标 expert 会降低样本效率，并使用“先学习路由、再蒸馏并冻结轻量 router”的两阶段方法缓解该问题。它来自语言模型，不是机器人控制证据，但说明必须测量 route drift，不能只看最终 usage。

[TS2Vec](https://ojs.aaai.org/index.php/AAAI/article/view/20881)通过不同上下文视图之间的层次对比学习获得 timestamp-level 表示，说明对局部时间窗口做多尺度编码是合理的。它适合作为后续自监督辅助目标，不足以证明对比损失应成为 PPO router 的主训练信号。

[Soft Modularization](https://proceedings.neurips.cc/paper/2020/hash/32cfdce9631d8c7906e8e9d6e68b514b-Abstract.html)展示了任务表示与状态共同调制模块组合的端到端方法，支持 soft/Top-K 路由和从控制目标学习模块分工；但其任务标识和 Meta-World 设置与匿名 motion window 不同。

这些工作共同支持两点：

- router 可以随任务训练，而不是必须离线标注；
- route drift 是要监控和抑制的优化问题，但不等于实际 gate 必须按 rollout 硬冻结。

### 2.3 为什么不把 SwAV/prototypes 放在第一版

[SwAV](https://proceedings.neurips.cc/paper/2020/hash/70feb62b69f16e0238f741fab228fec2-Abstract.html)证明了表示与 online cluster assignment 可以联合学习；[The Benefits of Balance](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f23653913d8390cd4fc1bee8a3238e17-Abstract-Conference.html)进一步说明，聚类中的平衡约束可以被理解为目标边缘分布假设与方差控制。

它们对本问题的正确启发是“需要防止 collapse”，而不是“真实运动类别必须均匀”：

- 当前数据中的走路、站立、跳跃和失败恢复很可能天然不等频；
- 本仓库当前 8 个 experts 全部参与前向计算，没有稀疏系统的容量溢出问题；
- 强制每个 prototype 接收相同样本数，可能把高频模式机械切碎，或把不同的稀有瞬态合并；
- HoloMotion 当前公开配置也没有使用 uniform load balance，而是只把使用不足的 expert 拉回 Top-K 竞争边界。

因此，SwAV/EMA prototypes 应是路由漂移或表示不足时的第二阶段选项，而不是首版前提。

## 3. 本仓库的事实基础

### 3.1 已有数据足够构造 reference-only router

[`spv5.py`](../src/sp_tracking/tasks/tracking/mdp/spv5.py)已经定义：

- reference input：`[-42, ..., +7]`，共 50 帧；
- 每帧：root position 3 维、root rotation 6D、29 个关节位置，共 38 维；
- 总输入：1900 维；
- clean supervision support：`[-3, ..., +7]`，共 11 帧、418 维。

在 50 Hz 下，该窗口覆盖约 0.84 秒历史与 0.14 秒未来。环境端在生成 noisy input 时已经取得 clean 50 帧，随后只把最后 11 帧暴露为 supervision target。这意味着：

- MVP 不需要新增 observation，也不需要 clean 50 帧；
- 若后续需要 full-window teacher target，环境端不需要再次查询 motion，但 rollout 存储会显著增加。

### 3.2 当前 router 实际看的是完整状态

[`residual_moe.py`](../src/sp_tracking/tasks/tracking/rl/residual_moe.py)中的当前路径是：

```text
1651 维完整 policy feature
    → context encoder
    → shared feature
    ├→ router
    └→ 8 个 residual experts
```

因此 current router 会同时响应 reference、proprioception、tracking error、robot key-body、估计 root/contact 等变化。它没有“reference 分类器”的结构约束，无法区分：

- 专家切换是因为目标运动阶段变化；
- 还是因为同一目标下机器人受到扰动或估计噪声变化。

这正是 reference-only baseline 必须存在的原因。

### 3.3 当前 PPO 已有可复用的 collect-level hook

[`ppo.py`](../src/sp_tracking/tasks/tracking/rl/ppo.py)中的 `SPV51ContactEstimatorMoEPPO` 已经：

- 对完整 rollout 的 dense routing probability 求平均；
- 计算其相对 uniform 分布的 KL；
- 以 chunked backward 方式回传，避免保留整段 rollout 图；
- 在每个 minibatch 计算 router entropy 和 Top-1 probability；
- 保存 router update count。

现有基础可以直接扩展路由诊断与 inactive-expert margin。需要注意：

- 当前 `balance KL = 0.01` 是负载正则，不是 motion 分类监督；
- `confidence loss = 0`，首版应继续保持关闭；
- 当前 rollout context cache 只冻结受监督 reference decoder、root estimator 和 contact estimator 的输出；它没有冻结 MoE route；
- 标准 PPO 依靠 old log-probability 与当前 policy 的 ratio 约束更新。强行复用 rollout 时的 hard expert ID 会改变策略前向语义，除非完整重写行为策略与学习策略的路由契约。

因此，第一版不应“冻结实际 hard route”。如果后续加入 EMA/prototype，只冻结其辅助 target；真实 PPO gate 仍按当前参数前向。

## 4. 推荐架构

### 4.1 输入规范化

router 只接收部署时可用的 noisy 50 帧 reference，不接收：

- robot proprioception；
- tracking error；
- robot key-body；
- estimated root state 或 contact；
- motion ID、文件名或人工类别。

在网络前做确定性的 reference-frame 规范化：

1. 以窗口中 `step = 0` 的 reference root 为锚点；
2. 所有 root XY 减去当前 reference root XY；
3. 所有平面位移和朝向旋转到当前 reference heading frame；
4. 保留绝对 root height、roll/pitch 和关节位置，因为它们包含下蹲、腾空、倾倒等信息；
5. 可增加相邻帧有限差分得到的 root/joint velocity，但应作为消融，避免首版扩大输入。

这样可保证 expert ID 不被世界坐标平移和全局朝向主导，同时保留决定局部动力学的高度、姿态与时间变化。

### 4.2 Temporal router

MVP 建议用轻量 TCN，而不是 Transformer 或 Periodic Autoencoder：

```text
noisy reference：50 × 38
    ↓ reference-frame canonicalization
    ↓ 独立 router normalizer
    ↓ frame projection：38 → 128
    ↓ 3 个 temporal residual Conv1d block
    ↓ temporal pooling / current-token readout
    ↓ 128 维 router embedding
    ↓ linear：128 → 8 logits
    ↓ temperature softmax
    ↓ Top-2 + selected-weight renormalization
```

选择 TCN 的理由：

- 50 帧窗口固定且较短；
- 卷积天然提取速度、加速度和局部相位；
- 参数与导出成本小，便于和当前 router 做等预算比较；
- DeepPhase、TS2Vec 和大量时间序列编码器都表明 temporal convolution 是合适的基本算子，但这里不继承它们的特定训练目标。

第一轮可从 128 维、3 个 block 开始；具体 kernel、dilation 和 pooling 属于工程超参数，不是文献结论。

### 4.3 Expert 路径保持完整状态

当前 shared context encoder、shared residual block、8 个 residual experts 和 action head均可保留。唯一结构变化是把 router 的输入与 expert 的输入解耦：

```text
policy feature 1651 ─→ context encoder ─→ shared block ───────────────┐
                                                        ├→ expert 0 ─┤
                                                        ├→ ...       ├→ Top-2 混合
                                                        └→ expert 7 ─┘

reference 50 × 38 ─→ reference router ─→ 8-way Top-2 weights ─────────┘
```

这个分解不意味着控制器忽略失衡状态。完整状态仍决定 shared feature 和每个 expert residual 的数值，只是不再允许状态扰动重新定义“当前属于哪类 reference motion”。

### 4.4 暂不加入 state correction

首版不加入 `q(k | reference, state)` 修正项。理由不是 state 永远无用，而是：

- HoloMotion 给出了 full-observation routing 导致实机 route oscillation 的直接反例；
- 当前目标首先是判断 reference 分类本身是否成立；
- reference prior 与 state correction 同时训练后，无法判断最终分类来自哪一侧；
- full state 已经进入专家函数，不需要通过改 expert ID 才能执行恢复动作。

只有当 reference-only 路由在强扰动下显著弱于当前 router，且失败可归因于“同一 reference 确实需要不同专家”时，才测试受限 correction，例如：

- correction logits 做有界缩放；
- 只使用 tracking-error/contact 的低维摘要；
- 以 reference logits 为主项；
- 同时报告 correction 的触发率和对 route 的改变量。

## 5. 训练目标与稳定化顺序

### 5.1 MVP：只让 Top-2 gate 通过 PPO 学习

第一版总目标保持简单：

```text
现有 PPO 目标
+ 现有 estimator / contact / reference reconstruction 目标
+ 一个防止 expert 永久死亡的路由正则
```

建议先做两个完全可比的分支：

- **B2-balance**：保留当前 rollout-level uniform balance KL，系数仍为 0.01；
- **B2-margin**：关闭 uniform balance，改用 inactive-expert margin。

不要同时启用 confidence entropy。路由从接近均匀逐步产生专长时，过早压低熵会放大随机初始化带来的 rich-get-richer。

### 5.2 inactive-expert margin

可参考 HoloMotion 的实现语义：

1. 在 rollout 上统计每个 expert 的 hard Top-K usage；
2. 把 usage 低于“最高 usage × ratio floor”的 expert 标记为 inactive；
3. 对 inactive expert，惩罚其 logit 低于当前样本第 K 名 logit 的差值；
4. 只把死亡 expert 拉回“有机会进入 Top-K”的边界，不要求所有 expert 等频。

该方法比 uniform KL 更符合当前目标：

- 防止随机早期永久饿死；
- 允许真实 motion 模式保持不均衡；
- 不把计算负载平衡误当作语义类别先验。

但不能直接复制 HoloMotion 的 `weight = 10`：其 1024-expert Transformer 的损失尺度与本仓库 8-expert residual MLP 不同，系数必须通过梯度范数和短程消融确定。

### 5.3 表示不足时再加入辅助目标

如果 B2 的路由只按静态姿态而不是动力学分组，再依次测试：

1. **clean support reconstruction**：从 router embedding 预测现有 11 帧 clean reference target 或其速度；
2. **two-view consistency**：对同一个 raw window 施加两次部署一致的噪声/遮挡，约束 dense route distribution 一致；
3. **EMA teacher**：teacher 只提供稳定的 soft route target，student gate 继续通过 PPO 更新；
4. **online prototypes**：仅在连续表示已经稳定、但 expert assignment 仍漂移时加入。

辅助目标必须满足：

- teacher 可以看 clean reference，实际 router 只能看 noisy deployable reference；
- 不使用 robot state 生成 motion 类别；
- 辅助损失不能绕过 PPO，单独决定实际 hard route；
- checkpoint 保存 teacher、prototype 和 update schedule 的全部状态。

### 5.4 不默认加入相邻帧 switch penalty

相邻窗口高度重叠，不代表 route 必须相同。起跳、着地和恢复的真实边界恰恰需要快速切换。HoloMotion 的当前公开配置也关闭了 adjacent switch penalty。

首版只记录：

- dense distribution 的相邻 JS divergence；
- Top-1/Top-2 switch rate；
- route dwell time；
- switch 与 reference velocity/acceleration/change score 的关系；
- episode reset 处必须 mask。

只有观察到“reference 几乎不变但 route 高频抖动”时，再使用边界感知一致性：reference change 越小，连续性权重越大；真实变点附近不惩罚。

## 6. 具体代码改造点

### 6.1 `residual_moe.py`

保留当前 `ObservationConditionedResidualMoE` 作为基线，新增或扩展一个输入解耦版本：

```python
forward(policy_value, router_reference)
routing_probabilities(router_reference)
```

其中：

- `policy_value` 只用于现有 context encoder、shared block 和 experts；
- `router_reference` reshape 为 `[..., 50, 38]`，经 canonicalizer、normalizer、TCN 得到 logits；
- route 权重之外不把 router embedding 拼入 shared feature，避免 A/B 结果混入额外条件信息；
- 保留 dense-compute Top-2，暂不优化真正的 sparse dispatch。

建议独立实现 `ReferenceWindowRouter`，暴露：

- `dense_probabilities(reference)`；
- `sparse_probabilities(reference)`；
- `embedding(reference)`；
- 可导出的 canonicalization。

### 6.2 `spv5_1_models.py`

`SPV51ContactEstimatorMoEActor` 需要覆盖 actor forward，而不能只改 `get_latent()`：

1. `get_latent(obs)` 继续返回 1651 维完整 policy feature；
2. 新增 `_router_reference(obs)`，读取 `reference_encoder_input_group`；
3. actor forward 同时调用 `self.mlp(policy_latent, router_reference)`；
4. `routing_probabilities(obs)` 使用相同的两个输入；
5. distribution 的 update、sample 和 deterministic output 语义保持与父类一致；
6. ONNX/JIT export 同样从现有扁平 observation 中切出 raw reference 并传给 router。

router 应有独立 normalizer。现有 `reference_input_normalizer` 服务于 flat reconstruction MLP；在 reference-frame canonicalization 后，其统计分布已改变，不应默认复用。新的 normalizer 需要在 `update_normalization()` 中更新，并进入 state dict。

### 6.3 `ppo.py`

在现有 collect-level hook 上增加：

- hard Top-K usage 与 dead expert ratio；
- inactive-expert margin；
- `[time, env, expert]` 形状的相邻 route 诊断；
- 用 `dones` 屏蔽 episode 边界；
- reference-change-conditioned switch 统计；
- 可选的 behavior route distribution 快照，用于 route drift 诊断。

不建议第一版把 behavior hard route 存入 storage 并强制学习阶段复用。若需要 PPO epoch 内稳定性，可先加当前 route 相对 behavior dense distribution 的小权重 KL 消融，并同时监控 policy KL；它比冻结离散 expert ID 更符合连续策略更新。

### 6.4 配置

建议新建配置而不是改变当前基线，至少包含：

```text
moe_router_source: reference_window
moe_router_encoder: tcn
moe_router_embedding_dim: 128
moe_router_canonicalize_xy_heading: true
moe_router_balance_mode: uniform_kl | inactive_margin | none
moe_router_confidence_loss_coef: 0.0
moe_router_temporal_loss_coef: 0.0
moe_router_aux_reconstruction_coef: 0.0
```

所有新行为默认显式配置，避免旧 checkpoint 因隐式默认值改变语义。

### 6.5 测试

在现有 [`test_residual_moe.py`](../tests/test_residual_moe.py)和
[`test_spv5_reference_encoder.py`](../tests/test_spv5_reference_encoder.py)基础上增加：

1. **平移不变性**：对全部 root XY 加同一偏移，route 不变；
2. **heading 不变性**：整体绕竖直轴旋转，规范化后 route 不变；
3. **状态隔离**：保持 reference 不变、任意改变 robot state，route 严格不变；
4. **reference 敏感性**：改变时间结构时 route/logits 会改变；
5. **Top-2 正确性**：恰有两个非零权重且归一化；
6. **inactive margin 梯度**：死亡 expert 的 logit 获得朝 Top-K 边界的梯度；
7. **reset masking**：episode 边界不计 temporal switch；
8. **normalizer/checkpoint**：保存恢复后 route 一致；
9. **ONNX/JIT parity**：导出前后 logits、Top-2 和动作一致。

“reference 敏感性”不应断言随机初始化必须产生某个固定 expert ID；测试应检查数值依赖和梯度，而不是给 expert 赋人工语义。

## 7. 实验矩阵

| ID | Router | 路由正则/辅助 | 目的 |
| --- | --- | --- | --- |
| B0 | 当前 full-observation router | 当前 balance KL | 仓库基线 |
| B1 | 只用当前短时 decoded/policy reference feature | 当前 balance KL | 判断长窗口是否必要 |
| B2 | 50 帧 TCN reference-only | 当前 balance KL | 主假设，最小改动 |
| B3 | 50 帧 TCN reference-only | inactive margin | 检验非均匀防坍缩 |
| B4 | B3 | clean support/dynamics auxiliary | 检验时间表示是否不足 |
| B5 | B4 | EMA teacher / prototypes | 只处理实测 route drift |
| B6 | B4 | 有界 state correction | 只检验强扰动恢复 |
| C0 | 等参数 dense MLP | 无 router | 区分 MoE 专长与单纯增参 |
| C1 | 随机平衡 Top-2 | 无学习 router | 区分学习路由与容量效应 |

第一批实验应只跑 B0、B1、B2、B3、C0。B4–B6 都依赖前一阶段观测，不应同时投入。

为控制变量：

- 相同 motion split、domain randomization、seed、rollout 数和优化预算；
- 至少 3 个 paired seeds，重要结论优先 5 个；
- 报告 actor 参数量、训练 FLOPs、推理延迟和显存；
- B1/B2 增加的 router encoder 参数应单独列出，C0 做等参数对照。

## 8. 评估与判定

### 8.1 控制性能

- train、held-out 和 OOD motion 的 success/fall rate；
- joint、key-body、root position/orientation/velocity error；
- 高动态 motion 与尾部样本，而不只报告全局均值；
- push、地面参数变化、reference noise 下的恢复；
- 动作平滑性、torque/功率和策略 KL。

### 8.2 路由稳定性

- dense entropy、Top-1 confidence、effective experts；
- hard usage、dead expert ratio、每类 motion 的 usage 分布；
- 相邻 JS、Top-1 switch、dwell time；
- 相同 reference、不同 robot perturbation 下的 route JS；
- 相同 clean window、不同 noisy view 下的一致性；
- 训练 checkpoint 间的 route drift。

Expert ID 具有置换对称性。跨 seed 或 checkpoint 比较时，必须先用 Hungarian matching 对齐 expert，再计算 assignment agreement、ARI/NMI 或 confusion matrix，不能直接比较原始编号。

### 8.3 专长是否真实存在

路由稳定不等于专家有用，需要做 counterfactual：

1. 固定 reference window 和 robot state；
2. 分别强制每个 expert 或每个 Top-2 组合；
3. 比较动作分布、短时 tracking error 与失败率；
4. 形成 `expert × reference-cluster` 性能矩阵；
5. 检查 router 选择是否与更优专家具有统计一致性。

梯度 affinity 可以作为补充：

- 同 route 样本在 shared/expert 层的梯度余弦；
- route 内与 route 间的 cancellation；
- router 改造后是否提高被共享样本的优化相容性。

梯度仍然是验证信号，不作为训练初期的 cluster ground truth。

### 8.4 建议的通过标准

在训练前写死非劣阈值和统计方法。方向成立至少应同时满足：

1. B2/B3 的 held-out tracking 不劣于 B0；
2. 同一 reference 下施加 robot perturbation，B2/B3 route 明显比 B0 稳定；
3. B2/B3 没有长期 dead experts，或 inactive margin 能恢复它们；
4. counterfactual 矩阵显示不同 experts 对不同窗口分组具有可重复的相对优势；
5. 以上结论跨 paired seeds 成立，而不是单次训练偶然现象。

如果控制性能改善但 expert ID 没有清晰人工语义，方向仍可能成立；MoE 需要的是可复用控制专长，不是可命名类别。

## 9. 结果解释与下一步

| 观测 | 更可能的解释 | 下一步 |
| --- | --- | --- |
| B2 优于 B0，route 更稳 | reference-only 假设成立 | 继续 B3/B4，暂不加 state correction |
| B2 路由稳定但性能下降 | window 表示或专家选择不充分 | 比较 B1、增加 dynamics auxiliary |
| B2 在 push 下弱于 B0 | 专家内部恢复能力不足，或确需状态改路由 | 先增强专家输入/训练，再做 B6 |
| usage collapse | 早期竞争失衡 | inactive margin、较软温度或短 warm-up |
| usage 均衡但无专长 | uniform KL 在制造配额而非类别 | 换 inactive margin，检查 counterfactual |
| 相邻帧频繁切换但 reference 变化大 | 可能是真实阶段边界 | 不加平滑，按 change score 分层分析 |
| 相同 reference 因 state 改变 route | 结构隔离失败 | 修正 actor/router 输入与导出路径 |
| checkpoint 间 ID 全交换 | permutation 或 routing fluctuation | 对齐后再判断；必要时加入 EMA teacher |

## 10. 不确定性与适用边界

已确认事实：

- 当前仓库具备 50 帧 noisy reference、8-expert Top-2 residual MoE 和 collect-level router hook；
- GMT 展示同一 motion 内的门控阶段切换；
- HoloMotion 论文和当前公开实现采用 reference-only routing，并报告 full-observation routing 的实机振荡问题；
- Humanoid-GPT 的 HME 是整段序列离线聚类，不是 online window classification。

基于证据的工程推断：

- reference-only TCN 是当前仓库成本最低、可解释性最强的下一基线；
- inactive-expert margin 比强制 uniform cluster 更符合 8-expert dense-compute 场景；
- state correction、EMA/prototypes 和 temporal penalty 都应由失败模式触发。

仍待确认：

- HoloMotion 的结论能否从 1024-expert Transformer 迁移到本仓库 8-expert residual MLP；
- 最优窗口长度、时间编码器和 expert 数；
- noisy 50 帧中的过去信息是否比当前及短未来更重要；
- reference-only expert identity 是否足以覆盖极端失衡恢复；
- learned assignment 是否形成跨训练稳定的控制模式。

## 11. 推荐实施顺序

1. 保留 B0，新增 B2 的 reference-only TCN router 与结构不变性测试；
2. 增加 route stability、state-isolation、switch 和 counterfactual 诊断；
3. 跑 B0/B1/B2/C0 paired seeds；
4. 若 collapse，比较当前 uniform KL 与 inactive margin，得到 B3；
5. 若路由忽略动力学，再加 clean support/dynamics auxiliary，得到 B4；
6. 只有 route drift 明显时加入 EMA/prototypes；
7. 只有强扰动反复证明 reference-only 不足时测试有界 state correction。

一句话总结：

> 对当前 SP Tracking，正确的第一步不是先造一个复杂的在线聚类系统，而是把 50 帧 ref motion window 变成唯一的 expert selector，让 PPO 在训练中直接学出 Top-2 类别；完整机器人状态继续驱动共享主干和专家动作，分类稳定性用 margin、诊断和必要时的辅助表示逐层补强。
