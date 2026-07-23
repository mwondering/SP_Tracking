# 面向大规模机器人运动跟踪的 PPO 探索增强方法调研

> 调研日期：2026-07-23
>
> 适用仓库：SP Tracking
>
> 关注范围：与 Split and Aggregate Policy Gradients（SAPG）相近，或能与
> PPO/SAPG 组合以提高探索多样性的方法。
>
> 状态：文献调研与工程建议；除仓库现有 SAPG 外，其余推荐方案尚未在本仓库实现或验证。

## 1. 结论摘要

当前最值得继续研究和移植的工作是 **Coupled Policy Optimization（CPO）**。这里的
CPO 指 ICLR 2026 的 Coupled Policy Optimization，不是同缩写的 Constrained Policy
Optimization。它直接继承 SAPG 的 leader–follower 和跨策略重要性采样框架，针对
SAPG follower 可能偏离 leader 过远、导致重要性权重失效的问题，用 KL 约束控制策略
距离，并用可选的对抗奖励防止多个 follower 重新塌缩为同一行为。其问题设定、代码
基础和实验平台都与本仓库最接近。

第二优先级是 **SAPG + generalized State-Dependent Exploration（gSDE）**。CPO
控制的是策略之间的多样性和样本可用性，gSDE 改善的是单个策略内部探索的时间
一致性；两者作用层面不同，可以组合。gSDE 对高频机器人控制尤其有吸引力，因为
逐控制步独立采样的高斯动作噪声可能形成抖动，却不一定带来有效的状态覆盖。

**Evolutionary Policy Optimization（EPO）** 是更激进的 SAPG 后继路线：将各
follower 的 latent embedding 当作基因，通过选择、交叉和变异显式产生策略分化，
再由 master 聚合群体经验。它适合在 CPO 诊断显示“各 block 仍然过于相似”时考虑，
但需要可靠的 episode fitness。对多动作跟踪而言，朴素平均回报可能偏向简单动作，
因此不能直接照搬。

不建议把 RND、DRND 或 E3B 作为本仓库的第一个探索增强实验。它们主要解决稀疏
奖励或状态覆盖问题，而当前任务具有密集跟踪奖励、参考动作条件和高维观测。
如果直接在完整 observation 上计算 novelty，模型可能奖励“不同的参考片段”，而
不是奖励更好的控制发现。此外，当前 SAPG 构造分支明确拒绝 RND，需要额外处理
内在奖励、critic、归一化和跨策略聚合语义。

推荐实施顺序如下：

1. 先补齐 SAPG 的 follower–leader KL、重要性比率偏离量、归一化有效样本量
   （ESS）和 block 级轨迹覆盖诊断；
2. 若 KL 高、ESS 低或 off-policy clipping 严重，先实现 CPO 的 KL/AWAC follower
   更新，不加对抗奖励；
3. 若策略距离合理但轨迹覆盖仍不足，再加入 CPO 对抗奖励或 gSDE；
4. 只有在确认现有 block 缺少显式分化时，才实验 EPO；
5. 只有在任务奖励确实稀疏或存在长时程发现瓶颈时，再引入 RND/E3B 类内在奖励。

以上排序是面向本仓库结构的工程判断，不是已有论文在 G1 多动作跟踪上的直接结论。
目前没有一篇所调研文献直接验证这些方法对本任务一定有效。

## 2. 问题定义与术语

“提高 PPO 探索性”至少包含三个不同问题，不能只用动作熵统一解释。

### 2.1 动作级随机性

策略在同一状态下是否会采样不同动作。常见方法是增大高斯标准差、增加 entropy
bonus，或使用状态相关噪声。动作熵提高并不保证长期轨迹变得多样；相邻时刻独立
噪声还可能互相抵消，只表现为高频抖动。

### 2.2 轨迹或状态访问多样性

策略是否到达不同状态、采用不同接触模式、恢复路径或局部控制策略。这比动作熵
更接近 SAPG 关注的问题。不同策略即使具有相同瞬时动作熵，也可能形成显著不同
的完整轨迹。

### 2.3 大规模并行数据的有效性

在一万到数万个同步环境中，更多样本不等于更多有效信息。如果所有环境都由同一
高斯策略采样，大量动作仍集中在均值附近，轨迹可能高度重复。反过来，如果 follower
与 leader 差异过大，虽然轨迹很多样，但这些 off-policy 样本会因重要性比率极端、
PPO clipping 或梯度高方差而无法有效训练 leader。

本文因此采用以下判断标准：

- **探索广度**：访问状态、接触模式和局部轨迹的覆盖范围；
- **探索质量**：探索样本对最终 leader 的学习是否有正贡献；
- **有效样本量**：重要性加权后，样本是否仍然具有足够权重；
- **稳定性**：探索增强是否导致 PPO clipping、梯度方差或 critic 误差恶化；
- **最终部署成本**：训练时的 population、判别器或噪声模块能否在导出时移除。

## 3. 当前仓库基线

本仓库已经实现 opt-in 的 SAPG 扩展，语义跟随作者官方 `rl_games` 实现，而不是
只借用“多策略”名称。当前实现具有以下性质：

- 环境按连续区间划分为若干 policy block；
- 最后一个 block 是 leader，其余 block 是 followers；
- actor 和 critic 共享主干，并通过各自的 learned policy embedding 条件化；
- 每个 block 有独立的 state-independent Gaussian 标准差；
- 所有原始轨迹保留为各策略的 on-policy 数据；
- 每轮随机选择 `off_policy_ratio` 个 follower，将其完整轨迹再次作为 leader
  数据；
- follower 行为 log-probability 保持冻结，leader 在聚合样本上重新计算当前
  policy/value；
- off-policy critic 使用官方的一步 bootstrap target；
- 普通 on-policy 样本仍使用 GAE；
- 导出时只保留 leader，并把 leader embedding 折叠进网络偏置。

对应说明和实现主要位于：

- [`sapg.md`](./sapg.md)
- [`sapg/config.py`](../src/sp_tracking/tasks/tracking/rl/sapg/config.py)
- [`sapg/conditioning.py`](../src/sp_tracking/tasks/tracking/rl/sapg/conditioning.py)
- [`sapg/batch.py`](../src/sp_tracking/tasks/tracking/rl/sapg/batch.py)
- [`sapg/update.py`](../src/sp_tracking/tasks/tracking/rl/sapg/update.py)
- [`sapg/extension.py`](../src/sp_tracking/tasks/tracking/rl/sapg/extension.py)

默认配置是 4 个 policy blocks、32 维 local embedding 和 1 个被聚合 follower。
`exploration_type: entropy` 可以给不同 block 分配从 0.5 到 0.0 线性递减的熵奖励
系数；leader 的该项为零。当前日志已经包含总 clip fraction、off-policy clip
fraction、被选 follower 和各 block 平均标准差，但尚未直接记录：

- leader 与每个 follower 的策略 KL；
- off-policy importance ratio 相对 1 的偏离程度；
- importance sampling 的归一化 ESS；
- 各 block 的 motion、状态或控制行为覆盖；
- 各 follower 样本对 leader 梯度的方向和实际贡献。

这些缺失指标决定了后续应该“增加多样性”还是“约束过度多样性”。在没有诊断的
情况下直接增大熵、增加 block 或加入内在奖励，可能得到更多互不相关、但对 leader
无效的样本。

当前 SAPG 还明确不支持 recurrent policy、RND、非 `GaussianDistribution` actor
和 `torch_compile_mode`。目前注册的 tracking tasks 使用 feed-forward Gaussian
policy，因此 CPO 和 gSDE 的结构改造具有可行性；RND 与 SAPG 的组合则需要解除
并重新定义现有限制。

## 4. 主要技术路线

### 4.1 多策略采样与经验聚合

代表工作是 SAPG 和 CPO。多个策略负责产生不同数据，但最终必须回答：其他策略的
数据如何安全地更新目标策略。SAPG 使用重要性采样和 PPO clipping 聚合 follower
数据；CPO 进一步控制 follower 与 leader 的距离，使被聚合数据既有差异又不过度
off-policy。

这条路线最适合：

- GPU simulator 提供一万以上并行环境；
- 最终只需要部署一个策略；
- 任务本身能用 PPO 稳定学习，但扩大 batch 后收益饱和；
- 主要瓶颈是并行数据重复，而不是完全没有外部奖励。

### 4.2 Population、进化与策略选择

代表工作是 DexPBT、EPO 和 Diversity via Determinants（DvD）。它们用多个策略或
latent 个体探索不同解，并通过超参数变异、基因变异、选择或群体多样性目标维持
差异。

这条路线能产生比简单熵奖励更长时程、更加一致的策略分化，但必须定义：

- 个体 fitness；
- 选择和变异周期；
- 如何避免群体只偏向容易任务；
- 被淘汰策略的数据是否浪费；
- 最终是部署最佳个体、master，还是蒸馏 population。

### 4.3 状态相关或参数空间探索

代表工作是 gSDE 和 Parameter Space Noise。它们不建立完整 population，而是让一段
时间内的探索扰动保持一致。相比逐步独立高斯噪声，这更有可能形成可辨识的动作
序列和状态迁移。

这条路线通常不改变外部奖励，也不要求额外 novelty critic，适合连续控制和机器人。
它解决的是单策略探索结构，不负责 SAPG follower 数据的 off-policy 可用性。

### 4.4 内在奖励与状态覆盖

代表工作包括 RND、Distributional RND（DRND）、RE3、RIDE、AGAC 和 E3B。它们
根据预测误差、状态熵、状态变化、策略不可预测性或回合内访问密度给予探索奖励。

这条路线在稀疏奖励、导航和 hard-exploration benchmark 上证据较强，但内在奖励
是否与任务能力一致取决于表示和奖励尺度。对参考条件运动跟踪，必须防止 novelty
主要由 reference ID、动作相位、随机化参数或传感器噪声贡献。

### 4.5 显式轨迹或策略多样性

TEEN、RSPO 和 DvD 直接优化状态–动作访问分布、轨迹新颖性或群体体积。这些方法
说明“多样轨迹”比“高动作熵”更接近有效探索，但其原始实现通常不是大规模同步
PPO，也没有 SAPG 的 leader 聚合语义。它们更适合作为设计探索度量和辅助损失的
思想来源。

## 5. 代表性工作分析

### 5.1 SAPG：Split and Aggregate Policy Gradients

**文献信息**

- Jayesh Singla、Ananye Agarwal、Deepak Pathak；
- ICML 2024 Oral；
- 官方论文、项目页和代码均公开。

**研究问题**

SAPG 研究的是：当 GPU simulator 能同步运行上万环境时，为什么 PPO 的最终性能
仍会随 batch size 饱和，以及怎样利用这些环境产生更有价值的数据。

**核心方法**

SAPG 把 N 个环境分给 M 个策略。策略共享 backbone，但用 local parameters 区分；
followers 只做自己的 on-policy 更新，leader 同时使用自己的 on-policy 数据和经
重要性采样修正的 follower 数据。leader–follower 非对称设计用于避免所有策略因
共享相同数据重新趋同。

**论文证据**

论文在 Isaac Gym 的灵巧手、手臂和双臂操作任务上比较 PPO、DexPBT 和 PQL，并
报告更高的样本效率或最终性能。论文还用 PCA reconstruction error 和 MLP
reconstruction error 分析访问状态的多样性。

**证据边界**

- 主要证据来自仿真 dexterous manipulation；
- 训练环境规模很大，收益不能自动外推到小规模环境；
- SAPG 只探索了有限数量的聚合方式和多样性机制；
- 原论文没有系统回答“策略应该相距多远”；
- 更高访问状态多样性不能单独证明每个 follower 样本对 leader 都有效。

**对本仓库的意义**

本仓库已复现其关键训练语义，因此 SAPG 应作为所有后继实验的共同基线，而不是
重新实现。下一步重点应该是测量聚合样本是否有效，而不是再次证明多策略能产生
不同数据。

### 5.2 CPO：Rethinking Policy Diversity in Ensemble Policy Gradient

**文献信息**

- Naoki Shitanda、Motoki Omura、Tatsuya Harada、Takayuki Osa；
- ICLR 2026 Poster；
- 算法全称 Coupled Policy Optimization；
- 官方代码直接基于 SAPG 的 `rl_games` 代码库。

**研究问题**

CPO 追问 SAPG 留下的关键问题：更大的 follower–leader 差异是否必然带来更好的
学习。论文的答案是否定的。策略差异可以增加状态覆盖，但过大差异会使 importance
ratio 偏离 1，降低有效样本量，并增加 PPO clipping bias 和更新不稳定性。

**核心方法**

CPO 保留 SAPG 的 leader 聚合更新，同时修改 follower：

1. 将 follower 更新写成“最大化自己的 advantage，同时限制 follower 到 leader
   的 KL”；
2. 使用类似 AWAC 的闭式目标近似该约束，让 follower 从 leader 轨迹学习；
3. 用温度参数控制吸引强度；
4. 可选加入 agent-identity discriminator，根据 state-action 对生成对抗奖励，
   防止所有 follower 过度集中；
5. 最终使 followers 围绕 leader 分布，而不是无界发散或全部重合。

这里的重要区别是：CPO 不是单纯减小所有策略之间的 KL。它保持 leader–follower
非对称结构，控制 follower 对 leader 的距离，同时用对抗项维持 follower 之间的
差异。

**论文证据**

论文使用 24,576 个并行环境，在 6 个 dexterous manipulation、2 个 gripper
manipulation 和 2 个 locomotion 任务上比较 PPO、PBT、SAPG 和 CPO，每项报告
5 个随机种子的均值与标准差。

论文报告：

- 在多数复杂 manipulation 任务上，CPO 的样本效率和最终性能优于 SAPG；
- 在 Regrasping 和 Throw 上没有观察到对 SAPG 的显著提升；
- 多项任务中，CPO 用约一半环境步数达到 SAPG 的最终性能；
- ShadowHand 的归一化 ESS 从 SAPG 的 0.0223 提高到弱 KL 约束设置的 0.763，
  更强约束下进一步提高；
- KL-only 增加约 24% wall-clock，加入对抗奖励后约增加 52%；这是论文报告的
  特定硬件和实现结果，不是本仓库的运行时保证。

**证据强度**

CPO 是本调研中对当前问题证据最强的工作：

- 正式接收于 ICLR 2026；
- 直接比较 SAPG；
- 使用相同大规模并行机器人环境；
- 提供 5 seeds、消融、KL、importance-ratio deviation 和 ESS 分析；
- 官方代码基于 SAPG，便于核对实现。

**局限**

- 环境数量和每个 block 的环境数量固定；
- KL 强度仍是人工超参数；
- 主要实验仍是仿真；
- 论文没有覆盖多参考动作 tracking 和 motion imbalance；
- full CPO 的判别器规模和 wall-clock 开销不低；
- follower 使用 leader 样本时，advantage/critic 语义的移植必须与本仓库现有
  one-step off-policy target、辅助损失和多 GPU 归约一致。

**对本仓库的判断**

CPO 是最低结构风险、最高证据优先级的下一步。建议先移植 KL/AWAC follower
目标，不加 discriminator；只有确认 follower 过度聚拢后再加入对抗奖励。

### 5.3 EPO：Evolutionary Policy Optimization

**文献信息**

- Jianren Wang、Yifan Su、Abhinav Gupta、Deepak Pathak；
- 2025 年 arXiv 预印本；
- 公开项目页和官方代码；
- 截至调研日期，项目页仍将其标为 preprint，因此本文不把它视为与 CPO 同等级
  的同行评议证据。

**研究问题**

EPO 认为 SAPG 虽然有多个 latent-conditioned followers，但缺少显式机制持续产生
高质量差异。进化算法擅长群体探索，却样本效率低；SAPG 可以把进化群体的经验
聚合给 master，从而结合二者。

**核心方法**

- 所有 agent 共享 actor–critic 参数；
- 每个 agent 使用独立 latent gene 条件化；
- 每个 agent 完整交互一个 episode 并计算 fitness；
- 周期性保留 elite、淘汰弱个体；
- 对 elite gene 做 crossover 和 mutation 生成新个体；
- master 用自己的 on-policy 数据和 population 的 SAPG off-policy 数据更新；
- 非 master agent 仍主要用自己的 on-policy 数据更新。

**论文证据**

论文覆盖 Allegro-Kuka manipulation、ANYmal、Unitree A1 Parkour 和 DeepMind
Control 等 8 个任务，报告 5 seeds，并与 PPO、SAPG、PBT、CEM-RL、SAC 和 PQL
比较。作者报告在困难 manipulation 和 parkour 上收益明显，并在最多 49,152 个
环境下观察到继续扩展的趋势。官方复现实验常使用 24,576 个环境和 64 个 agent，
即每个 agent 约 384 个环境。

**局限**

- 主要证据来自预印本作者实验；
- fitness 是任务相关设计，可能成为主要收益或失败来源；
- selection、mutation、crossover 和更新周期引入更多超参数；
- 64 个 agents 与本仓库默认 4 blocks 差距较大；
- 对多动作数据直接使用全局平均 reward，容易选择擅长简单动作的 gene；
- 进化导致的突然 gene 变化可能破坏 checkpoint 连续性和现有辅助模型训练；
- 论文没有证明收益来自进化本身而非更大的 population 和任务特定 fitness 的
  全部交互。

**对本仓库的判断**

EPO 更适合作为第二阶段研究方向。若实现，fitness 至少需要按 motion group 或难度
分层，可能采用：

- 每个 motion group 的标准化 return 分位数；
- 对最差若干 motion 的 CVaR 型指标；
- tracking success 与 fall rate 的组合；
- 先在固定 motion subset 上配对比较，再扩展到全数据集。

不能只按全局平均 episode reward 淘汰 gene。

### 5.4 DexPBT：Population-Based Training

**文献信息**

- Aleksei Petrenko、Arthur Allshire、Gavriel State、Ankur Handa、
  Viktor Makoviychuk；
- 2023 年预印本；
- 与 SAPG 使用相同一组 Allegro-Kuka 等操作任务。

**核心方法**

DexPBT 在多个 PPO agent 上使用不同超参数并行训练，周期性淘汰低性能 agent，
复制高性能权重并变异超参数。其主要贡献是把 decentralized PBT 扩展到大规模
Isaac Gym dexterous manipulation。

**与 SAPG 的差异**

- DexPBT 的多样性主要来自超参数和独立训练；
- SAPG 的多样性来自策略条件化和不同探索强度；
- DexPBT 通常不让最终最佳策略学习所有其他 agent 的轨迹；
- SAPG 通过重要性采样聚合 follower 数据，减少失败 population 的数据浪费。

**适用判断**

如果真正不确定的是 entropy coefficient、学习率、reward weight 或 curriculum
速度，PBT 仍有价值；如果问题是大 batch 中轨迹重复，SAPG/CPO 更直接。在当前
仓库中，完整 PBT 还意味着管理多套 optimizer、checkpoint 和选择状态，工程成本
高于 CPO。

### 5.5 gSDE：Smooth Exploration for Robotic RL

**文献信息**

- Antonin Raffin、Jens Kober、Freek Stulp；
- CoRL 2022；
- 方法名 generalized State-Dependent Exploration；
- 代码进入 Stable-Baselines3 体系。

**研究问题**

连续控制通常每一步独立采样高斯动作噪声。在高控制频率机器人上，这会造成抖动、
磨损和较差探索：连续噪声可能经过物理系统低通后互相抵消，并不形成一致行为。

**核心方法**

gSDE 从 policy latent features 生成状态相关探索函数，并每隔 n 步重新采样该函数
的参数。n 在逐步独立噪声和整回合固定噪声之间提供连续折中。对于具有显式概率
分布的 PPO，可以用 gSDE 分布替换普通对角高斯，同时保持可计算的 log-probability。

**论文证据**

- 主要仿真实验以 SAC 为主，并在附录/消融中报告 PPO；
- PPO Walker2D 实验显示噪声重新采样周期对性能重要；
- 使用 policy latent 通常优于直接使用原始状态，尤其是 PPO；
- 实机结果使用 SAC + gSDE，在 tendon robot、quadruped 和 RC car 上证明平滑
  探索可用于直接真实机器人训练；
- 因此，论文直接支持“gSDE 与 PPO 兼容”和“机器人探索更平滑”，但不直接支持
  “PPO + gSDE 已在本文三种实机上验证”。

**对本仓库的判断**

gSDE 与 SAPG 正交且部署友好，但会改变当前 `GaussianDistribution` 的参数化和
log-probability 计算。必须保证：

- rollout 保存生成行为动作所需的噪声状态；
- follower behavior log-probability 可被严格重建；
- leader 对 follower 轨迹重新求值时使用 leader 的 gSDE 分布；
- 噪声重采样周期与 50 Hz 控制、rollout horizon 和 episode reset 对齐；
- 导出时禁用探索噪声，不增加部署输入。

若不能保证这些语义，先做普通 PPO + gSDE，而不是直接与 SAPG 组合。

### 5.6 Parameter Space Noise

**文献信息**

- Matthias Plappert 等；
- ICLR 2018；
- 实验涵盖 DQN、DDPG 和 TRPO。

**核心方法**

在 episode 开始时扰动 policy parameters，而不是每步扰动动作。参数扰动会在整段
轨迹中形成一致行为，噪声尺度根据扰动前后策略在动作空间的距离自适应。

**证据边界**

论文显示其在部分 sparse continuous control 和 TRPO 任务上优于动作噪声，但没有
直接实验 PPO，也明确指出参数噪声并不保证在所有任务提高探索。

**对本仓库的判断**

它可以转化为“每个 SAPG block 使用一组固定或慢更新的低秩参数扰动”，概念上比
完整 EPO 简单。但直接扰动大网络参数会与 learned local embedding、normalization
和 shared backbone 更新耦合。更稳妥的实现是仅扰动 local embedding 或小型
adapter，而不是扰动整个 actor。

### 5.7 RND、DRND、RE3 与 E3B

#### RND

Random Network Distillation 使用固定随机 target network 和可训练 predictor；
预测误差作为状态 novelty reward。它实现简单，在 hard-exploration Atari 上有强
结果，但 novelty 容易受随机观测、表示尺度和长期 predictor 遗忘影响。

#### DRND

Distributional RND 蒸馏随机网络分布，尝试让 bonus 更接近 pseudo-count 并缓解
RND 的 bonus inconsistency。ICML 2024 论文提供 PPO Atari 比较和公开代码，但
没有提供与本任务相近的连续机器人 tracking 证据。

#### RE3

Random Encoders for Efficient Exploration 使用固定随机 encoder，在低维表示中用
k-nearest-neighbor 估计状态熵。ICML 2021 论文在 locomotion、navigation 和
reward-free downstream task 上显示收益。它减少了训练表征漂移，但 kNN 计算和
跨大量并行环境的数据结构需要额外设计。

#### E3B

Exploration via Elliptical Episodic Bonuses 用 inverse-dynamics representation 表示
可控状态，并根据同一 episode 已访问 embedding 所张成的椭球计算 novelty。它比
离散 count 更适合连续状态，在 MiniHack 和 Habitat reward-free exploration 上有
较强证据；其 Habitat base algorithm 是 DD-PPO。但论文并未验证高频 humanoid
tracking。

#### 对本仓库的共同风险

完整 tracking observation 同时包含机器人状态、参考状态、未来 reference、tracking
error、估计接触和可能的 privileged information。novelty 模块若不分离这些成分，
可能主要响应：

- 新 motion 或新 phase；
- domain randomization 参数；
- push/noise；
- reference encoder 误差；
- 失败后偏离训练分布的异常状态。

前三类不一定代表应被奖励的控制发现，最后一类甚至可能鼓励摔倒和失控。因此如果
以后引入内在奖励，优先考虑只在可控 robot-state embedding、接触模式或恢复状态
上计算，并独立归一化 intrinsic/extrinsic returns。

### 5.8 TEEN、DvD、RSPO、AGAC 与 RIDE

这些工作不建议原样移植，但提供了有用设计思想：

- **TEEN，NeurIPS 2023**：用 policy ID discriminator 近似最大化不同 TD3
  sub-policies 的 state-action visitation discrepancy。说明动作差异应通过轨迹
  访问分布衡量；原方法基于 off-policy TD3 ensemble，不是 PPO/SAPG。
- **DvD，NeurIPS 2020**：用 behavioral embedding 的 determinant 衡量整个
  population 占据的体积，并在线调节 reward–diversity 权衡。适合借鉴群体多样性
  指标，但原方法和当前共享网络 leader 聚合结构不同。
- **RSPO，ICLR 2022**：根据轨迹相对既有策略是否新颖，在 extrinsic reward 和
  intrinsic diversity reward 间切换，迭代发现多个局部最优策略。目标是得到策略
  集合，不是训练单一 leader。
- **AGAC，ICLR 2021**：训练 adversary 模仿 actor，actor 通过与 adversary 的
  KL 获得“难以预测”奖励。在 procedurally generated hard-exploration 任务上有
  证据，但连续机器人控制证据不足。
- **RIDE，ICLR 2020**：奖励 agent 在 learned latent space 中引起的状态变化，并
  配合 episodic visitation count。适合交互式环境；对 tracking，单纯大幅改变状态
  可能与准确跟踪目标冲突。

## 6. 方法对比

| 方法 | 主要探索层面 | 是否直接基于 PPO/SAPG | 是否聚合其他策略数据 | 主要证据场景 | 对当前仓库适配度 |
| --- | --- | --- | --- | --- | --- |
| SAPG | 策略间轨迹多样性 | 是 | 是，importance sampling | 大规模 Isaac Gym manipulation | 已实现 |
| CPO | 受控策略多样性与 ESS | 是，直接扩展 SAPG | 是 | 10 个大规模机器人任务 | 最高 |
| EPO | 进化式 population diversity | 是，SAPG + GA | 是，由 master 聚合 | manipulation、locomotion、parkour | 高，但复杂 |
| DexPBT | 超参数与独立策略 population | PPO population | 通常不聚合全部轨迹 | Isaac Gym manipulation | 中 |
| gSDE | 单策略时间一致动作探索 | 与 PPO 兼容 | 否 | PyBullet、实机机器人 | 高，正交 |
| Parameter Noise | 参数空间长时程探索 | 论文验证 TRPO 等 | 否 | sparse continuous control | 中 |
| RND/DRND | 全局状态 novelty | 可与 PPO 组合 | 否 | Atari hard exploration | 中低 |
| RE3/E3B | 状态熵或回合内覆盖 | E3B 使用 DD-PPO | 否 | navigation、Habitat、MiniHack | 中低 |
| TEEN/DvD/RSPO | 显式轨迹或群体多样性 | 原实现不是 SAPG | 各异 | MuJoCo、策略发现 | 思想参考 |

## 7. 证据强度与不可直接外推之处

### 7.1 已有证据直接支持的结论

- SAPG 在一万级并行环境中使用多策略采样和经验聚合，能缓解单策略 PPO 数据
  重复问题；
- CPO 证明并实验验证了“更多策略差异不总是更好”，并把 KL、importance ratio、
  ESS 和 PPO clipping 联系起来；
- EPO 表明在作者实验中，显式进化 latent population 可以进一步扩大 SAPG 的
  策略搜索；
- gSDE 能使连续控制探索更加时间一致，并与 PPO 的概率策略形式兼容；
- novelty bonus 在多类稀疏奖励 benchmark 中能显著改善探索。

### 7.2 基于仓库结构的合理推断

- 当前仓库复用了 SAPG leader–follower、policy conditioning 和 aggregate batch，
  因而 CPO 的移植路径比 EPO、PBT 或 RND 更短；
- 当前 `sapg/off_policy_clip_fraction` 已能提供 CPO 动机的一部分证据，但只有
  加入 KL 和 ESS 才能区分“策略差异过大”和“同一策略 PPO 更新过大”；
- G1 tracking 的逐步高斯探索可能产生高频关节扰动，gSDE 可能比单纯提高 entropy
  更有效；
- EPO 的全局 fitness 可能偏向简单 motions，必须使用分层或风险敏感 fitness；
- 原始 observation novelty 容易把 reference diversity 当作 agent exploration。

### 7.3 仍未确定

- 当前 SAPG 在本任务上是否真的提高了 state/trajectory coverage；
- follower 是否已经发生严重 leader misalignment；
- off-policy 样本的低贡献主要来自 KL、advantage target、critic target，还是
  motion distribution mismatch；
- gSDE 是否会改善复杂 tracking，还是只降低动作抖动；
- CPO 的对抗奖励是否会发现有用恢复行为，还是破坏参考跟踪；
- EPO 的最佳 population size 和 fitness 定义；
- 任一方法对 sim-to-real 鲁棒性和部署稳定性的真实影响。

这些问题必须由配对实验回答，不能从 manipulation、navigation 或 Atari 结果直接
推出。

## 8. 建议先补充的诊断

### 8.1 Follower–leader KL

在同一批 follower 访问状态上，计算每个 follower 行为分布到 leader 分布的 forward
KL，同时也可记录 reverse KL。对当前对角高斯策略可以直接从均值和标准差计算，
无需额外环境交互。

建议日志：

```text
sapg/policy_kl/follower_<id>_to_leader_mean
sapg/policy_kl/follower_<id>_to_leader_p95
sapg/policy_kl/leader_to_follower_<id>_mean
```

只记录全局平均值可能掩盖少量严重失配状态，因此至少保留 mean 和 p95。

### 8.2 Importance-ratio deviation

对 leader 在 follower 样本上的重要性比率 w，记录：

```text
mean_abs_deviation = mean(|w - 1|)
ratio_p01, ratio_p50, ratio_p99
```

应在 PPO epoch 0、任何 optimizer step 之前记录一次“纯行为失配”，再分别记录
各 epoch 更新后的比率。否则 follower–leader 差异和本轮 PPO policy update 会混在
一起。

### 8.3 归一化 ESS

对非负 importance weights，可采用：

```text
ESS = (sum(w))² / sum(w²)
normalized_ESS = ESS / number_of_samples
```

若 PPO clipping 后的权重被用于实际目标，还应同时报告 raw-ratio ESS 和
clipped-ratio ESS。ESS 接近 1 表示权重较均匀；接近 0 表示少量样本支配估计。
必须按 follower 分开记录，不能只看 aggregate 总量。

### 8.4 轨迹多样性

不建议直接用完整 observation 的欧氏距离。优先使用任务相关但不含 motion ID 的
低维统计，例如：

- pelvis 线速度和角速度；
- feet/hand 接触 bit pattern 和切换频率；
- root height、姿态误差和 fall/recovery event；
- action smoothness、joint velocity、torque；
- reference-relative tracking error 的时间序列；
- 现有 reference/task embedding 中去除 policy ID 后的覆盖。

可以比较 block 内和 block 间的平均距离，或计算固定 encoder 表示上的 kNN coverage。
encoder 必须冻结，否则表示漂移会伪造 coverage 变化。

### 8.5 Block 级任务质量

每个 block 至少记录：

- episode return；
- tracking term；
- fall rate；
- motion group 或难度分位数；
- episode length；
- action std 和 entropy；
- 被 leader 聚合后的 actor gradient norm、与 leader on-policy gradient cosine。

这样才能判断 follower 是“探索了 leader 未发现的有效行为”，还是仅仅在困难或
失败状态中发散。

## 9. 推荐实施路线

### 阶段 0：纯诊断，不改变训练目标

新增 KL、importance-ratio deviation、ESS 和 block 级 coverage。保持现有 SAPG
算法、随机种子和总环境数不变。

至少比较：

1. PPO；
2. SAPG，4 blocks，`exploration_type: none`；
3. SAPG，4 blocks，`exploration_type: entropy`；
4. SAPG，8 blocks，保持总环境数不变。

核心问题不是谁的 return 最高，而是：

- 增加 blocks 是否真的增加 coverage；
- coverage 增加是否伴随 ESS 崩溃；
- entropy 是否让 follower 远离 leader；
- off-policy sample 对 leader gradient 是否仍有正向贡献。

### 阶段 1：CPO-KL/AWAC，不加对抗奖励

新增一个与 `compatibility: official` 并列的明确算法模式，不应悄悄改变现有 SAPG
语义。建议配置形态：

```yaml
sapg_cfg:
  enabled: true
  method: cpo
  follower_awac_coef: 0.001
  follower_temperature: 0.5
  adversarial_reward_coef: 0.0
```

配置名仅表示建议，最终应以本仓库 dataclass 命名规则为准。

第一轮只实现：

- leader 保持现有 SAPG update；
- follower 保持自己的 on-policy PPO；
- follower 额外从 leader samples 学习 AWAC 型约束目标；
- 保持所有 SPV/HEFT auxiliary objectives 只作用于原始 on-policy 样本；
- 新增 CPO loss、KL、ESS 和每类样本 mask 日志；
- checkpoint 保存新超参数和必要 RNG 状态；
- export 仍只导出 leader。

验收标准：

- 相比 SAPG，off-policy mean |w−1| 下降；
- normalized ESS 上升；
- off-policy clip fraction 下降或至少不再恶化；
- coverage 不塌缩到 PPO 水平；
- return、fall rate 和困难 motion 分位数不下降；
- 多 GPU 与单 GPU 样本 mask、loss 权重语义一致。

### 阶段 2：CPO adversarial reward

只有出现以下情况才进入该阶段：

- CPO-KL 的 ESS 明显提高；
- follower–leader KL 受控；
- 但 follower–follower 距离、coverage 或 block 行为多样性显著降低。

判别器输入不宜直接使用完整 privileged observation。优先从可部署或物理可解释的
state-action 子集开始，避免判别器只识别 motion ID、reference phase 或 domain
randomization。建议消融：

1. robot state + action；
2. reference-relative error + action；
3. frozen behavior embedding + action。

必须确认判别准确率不是靠 block-specific normalization、policy embedding 泄漏或
环境 ID 泄漏获得。

### 阶段 3：普通 PPO + gSDE

先独立验证 gSDE，不与 SAPG 同时修改。比较：

- 普通 Gaussian PPO；
- 相同初始 action variance 的 PPO + gSDE；
- 不同 noise resample intervals。

主要指标除 return 外还包括：

- action continuity；
- joint acceleration；
- torque variation；
- fall/recovery；
- trajectory coverage；
- deterministic evaluation performance。

若普通 PPO + gSDE 明确有效，再处理 SAPG behavior log-probability、block noise
state 和 leader re-evaluation 的组合语义。

### 阶段 4：EPO

只有在以下证据同时存在时考虑：

- SAPG/CPO 的各 follower 长期过于相似；
- gSDE 只能提高局部探索，不能发现新的高回报行为；
- 已有稳定的 motion-balanced fitness；
- 训练预算允许更大 population 和更多消融。

第一版只进化 local embedding，不交叉或扰动 shared network weights。先使用较小
population，如 8 或 16，验证 selection 是否真正改善 coverage 和困难 motion，
再评估论文常用的 64 agents。

### 阶段 5：内在奖励

只在发现下列明确瓶颈后使用：

- 外部奖励长时间为零或几乎无区分度；
- agent 无法接触某类状态或完成某个先决动作；
- 单纯 population diversity 不能发现该行为。

第一版应把 novelty 建在冻结的 controllable robot-state embedding 上，并做到：

- intrinsic/extrinsic reward 分开归一化；
- 独立 value head 或明确验证共享 critic 不产生偏差；
- leader 与 follower 的 intrinsic reward 语义一致；
- 评估时完全关闭 intrinsic reward；
- 单独报告 novelty、task reward 和失败状态访问率。

## 10. 最小实验矩阵

建议先在一个代表性任务和固定 motion subset 上完成下表，再扩展到全数据集。

| 实验 | Blocks | 探索机制 | 聚合机制 | 目的 |
| --- | ---: | --- | --- | --- |
| PPO | 1 | 原始 Gaussian | 无 | 基线 |
| SAPG-none | 4 | 独立 embedding/std | 原始 SAPG | 当前核心基线 |
| SAPG-entropy | 4 | 分 block 熵系数 | 原始 SAPG | 判断熵是否扩大有效覆盖 |
| SAPG-8 | 8 | 独立 embedding/std | 原始 SAPG | 判断更多 policies 的作用 |
| CPO-KL-weak | 4 | 受控 policy diversity | SAPG + follower AWAC | 判断 ESS 修复 |
| CPO-KL-strong | 4 | 更强 leader coupling | SAPG + follower AWAC | 探索–稳定性边界 |
| CPO-full | 4 | KL + adversarial reward | CPO | 防止 follower 聚拢 |
| PPO-gSDE | 1 | 状态相关时间一致噪声 | 无 | 独立验证单策略探索 |

所有实验应：

- 使用相同环境总数、rollout horizon 和训练环境步数；
- 至少使用多个 paired seeds；
- 同时报告同 iteration 和同 environment steps；
- 如果 wall-clock 差异明显，再报告同 wall-clock；
- 不只选择最终最好 checkpoint，应报告学习曲线和固定窗口统计；
- 对 easy/hard motions 分层报告，不只给全局均值。

## 11. 代码改造落点

### 11.1 `sapg/config.py`

增加算法模式、CPO follower 温度/权重、可选 adversarial reward 和诊断开关。所有
新字段应进入 checkpoint config equality，避免旧 checkpoint 被静默按新语义加载。

### 11.2 `sapg/batch.py`

CPO follower 需要消费 leader samples，因此 aggregate batch 必须显式区分：

- 原始 on-policy 样本；
- follower 复制给 leader 的 SAPG 样本；
- leader 提供给 follower 的 CPO constraint 样本；
- adversarial reward/discriminator 样本。

不能只依赖单一 `off_policy_mask`；建议使用不互斥的 typed masks 或 sample role。

### 11.3 `sapg/update.py`

这里是主要 actor objective 改造点：

- 保留现有 PPO surrogate；
- 增加 follower AWAC/CPO 项；
- 保持辅助监督 loss 的 on-policy 权重不变；
- 增加 KL、ratio deviation、ESS 和 role-specific gradient/logging；
- 避免把 duplicated samples 再次放大 symmetry、RMA、reference encoder 或
  estimator loss。

### 11.4 `sapg/conditioning.py`

CPO 可以复用现有 policy context。若引入 gSDE，需要扩展 actor distribution 而不是
在 context 中临时扰动输出，否则 behavior log-probability 和 KL 难以严格定义。

### 11.5 `sapg/extension.py`

构造阶段需要：

- 按 method 创建可选 discriminator；
- 将 discriminator optimizer/checkpoint 与主 optimizer 明确分离；
- 保持 SAPG-disabled 分支零行为变化；
- 继续拒绝尚未定义组合语义的 RND/gSDE 配置；
- 在组合方法实现并测试后再逐项解除限制。

### 11.6 测试

至少新增：

- CPO sample-role 和 mask 精确计数；
- follower AWAC loss 的手工小张量对照；
- KL/ESS 指标对已知高斯分布的解析值测试；
- duplicated samples 不改变 auxiliary loss 总权重；
- 多 GPU role selection 和 loss 归约一致性；
- checkpoint round-trip；
- SAPG-disabled、SAPG-official 和 CPO 三条路径互不污染；
- actor export 仍等于 leader deterministic output。

## 12. 风险清单

| 风险 | 表现 | 主要诊断 | 缓解方式 |
| --- | --- | --- | --- |
| Follower 过远 | 高 KL、高 clipping、低 ESS | policy KL、ratio p99、ESS | CPO follower constraint |
| Follower 过近 | coverage 接近 PPO | block 间行为距离 | adversarial reward、gSDE、EPO |
| 简单 motion 主导 | 平均回报升、hard motion 下降 | 分 motion 分位数 | 分层 fitness/采样 |
| 内在奖励错位 | novelty 升、tracking/fall 恶化 | reward 分解、失败访问率 | controllable embedding、reward clipping |
| 判别器信息泄漏 | 轻易识别 policy ID | 输入消融、打乱 reference | 移除 ID/privileged 特征 |
| Off-policy critic 偏差 | actor 看似改善、value error 发散 | role-specific value loss | target 消融、独立 critic 检查 |
| 计算开销掩盖收益 | env steps 改善但 wall-clock 变差 | wall-clock、GPU utilization | 先 KL-only、后 full CPO |
| 多方法同时修改 | 无法归因收益 | 消融矩阵 | 分阶段单因素引入 |

## 13. 最终建议

对本仓库最合理的研究主线不是笼统地“继续增加 PPO 的熵”，而是：

```text
先测量 SAPG 的策略距离与样本有效性
    ↓
若 follower 过远：CPO-KL/AWAC
    ↓
若 follower 过近：CPO adversarial reward 或 gSDE
    ↓
若仍缺少全局策略分化：EPO latent evolution
    ↓
若存在真正稀疏发现瓶颈：受控表示上的 intrinsic reward
```

第一项可实施工作应是给当前 SAPG 增加 KL、importance-ratio deviation 和 ESS。
如果这些指标已经健康，而 SAPG 仍未改善 tracking，则问题可能不是探索不足，而是
reward、motion sampling、critic target、梯度冲突或模型容量；此时继续加入探索
机制不会解决根因。

## 14. 参考文献与资源

1. Singla, J., Agarwal, A., & Pathak, D. (2024). **SAPG: Split and
   Aggregate Policy Gradients**. ICML 2024.
   [PMLR](https://proceedings.mlr.press/v235/singla24a.html) ·
   [项目页](https://sapg-rl.github.io/) ·
   [代码](https://github.com/jayeshs999/sapg)
2. Shitanda, N., Omura, M., Harada, T., & Osa, T. (2026).
   **Rethinking Policy Diversity in Ensemble Policy Gradient in Large-Scale
   Reinforcement Learning**. ICLR 2026.
   [OpenReview](https://openreview.net/forum?id=gRdEcs0qdd) ·
   [arXiv](https://arxiv.org/abs/2603.01741) ·
   [代码](https://github.com/Naoki04/paper-cpo-code)
3. Wang, J., Su, Y., Gupta, A., & Pathak, D. (2025).
   **Evolutionary Policy Optimization**. arXiv preprint.
   [arXiv](https://arxiv.org/abs/2503.19037) ·
   [项目页](https://yifansu1301.github.io/EPO/) ·
   [代码](https://github.com/YifanSu1301/EPO)
4. Petrenko, A., Allshire, A., State, G., Handa, A., & Makoviychuk, V.
   (2023). **DexPBT: Scaling up Dexterous Manipulation for Hand-Arm Systems
   with Population Based Training**.
   [arXiv](https://arxiv.org/abs/2305.12127)
5. Raffin, A., Kober, J., & Stulp, F. (2022).
   **Smooth Exploration for Robotic Reinforcement Learning**. CoRL.
   [arXiv](https://arxiv.org/abs/2005.05719)
6. Plappert, M. et al. (2018). **Parameter Space Noise for Exploration**.
   ICLR 2018.
   [arXiv](https://arxiv.org/abs/1706.01905)
7. Burda, Y., Edwards, H., Storkey, A., & Klimov, O. (2018).
   **Exploration by Random Network Distillation**.
   [arXiv](https://arxiv.org/abs/1810.12894)
8. Yang, K., Tao, J., Lyu, J., & Li, X. (2024).
   **Exploration and Anti-Exploration with Distributional Random Network
   Distillation**. ICML 2024.
   [OpenReview](https://openreview.net/forum?id=rIrpzmqRBk) ·
   [代码](https://github.com/yk7333/DRND)
9. Seo, Y. et al. (2021).
   **State Entropy Maximization with Random Encoders for Efficient
   Exploration**. ICML 2021.
   [PMLR](https://proceedings.mlr.press/v139/seo21a.html)
10. Henaff, M., Raileanu, R., Jiang, M., & Rocktäschel, T. (2022).
    **Exploration via Elliptical Episodic Bonuses**. NeurIPS 2022.
    [OpenReview](https://openreview.net/forum?id=Xg-yZos9qJQ) ·
    [arXiv](https://arxiv.org/abs/2210.05805)
11. Li, C., Gong, C., He, Q., & Hou, X. (2023).
    **Keep Various Trajectories: Promoting Exploration of Ensemble Policies
    in Continuous Control**. NeurIPS 2023.
    [arXiv](https://arxiv.org/abs/2310.11138)
12. Parker-Holder, J., Pacchiano, A., Choromanski, K., & Roberts, S. (2020).
    **Effective Diversity in Population Based Reinforcement Learning**.
    NeurIPS 2020.
    [论文页](https://proceedings.neurips.cc/paper/2020/hash/d1dc3a8270a6f9394f88847d7f0050cf-Abstract.html)
13. Zhou, Z., Fu, W., Zhang, B., & Wu, Y. (2022).
    **Continuously Discovering Novel Strategies via Reward-Switching Policy
    Optimization**. ICLR 2022.
    [OpenReview](https://openreview.net/forum?id=hcQHRHKfN_) ·
    [arXiv](https://arxiv.org/abs/2204.02246)
14. Flet-Berliac, Y. et al. (2021).
    **Adversarially Guided Actor-Critic**. ICLR 2021.
    [OpenReview](https://openreview.net/forum?id=_mQp5cr_iNy) ·
    [arXiv](https://arxiv.org/abs/2102.04376)
15. Raileanu, R., & Rocktäschel, T. (2020).
    **RIDE: Rewarding Impact-Driven Exploration for Procedurally-Generated
    Environments**. ICLR 2020.
    [OpenReview](https://openreview.net/forum?id=EnFsMkZWiQ)
