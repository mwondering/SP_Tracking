# PMOE：PAE 原型路由的 SPV5-2 MoE

PMOE 在 SPV5-2 actor 上增加 Periodic Autoencoder（PAE）、在线
K-Means prototypes 和 Top-2 residual MoE。公开任务 ID 为：

```text
SPTracking-G1-BFM-SPV5-2PMoEActor-HEFTCritic-HEFTReward
```

训练示例：

```bash
uv run sp-train \
  task_id=SPTracking-G1-BFM-SPV5-2PMoEActor-HEFTCritic-HEFTReward \
  motion_path=/path/to/motions
```

## 数据流

```text
noisy reference window：50 × 38
  │
  ├─ 去除首帧 root XY 与 heading
  ├─ PAE encoder
  ├─ [log(1 + amplitude), frequency, offset]：8 × 3
  ├─ 与 8 个在线 prototypes 计算标准化距离
  └─ soft nearest-prototype probability ── detach ── Top-2 route
                                                     │
完整 SPV5-2 policy feature ─ shared block ─ 8 experts ┴─ action
```

PAE 的 phase 参与运动窗口重建，但不进入聚类特征。这样同一种运动在
不同周期相位上可以共享 prototype；幅值、频率和 offset 则用于描述运动
强度、节律与姿态基线。

PAE 重建的是现有的 noisy 50 帧 reference input。本任务没有新增 clean
50 帧监督 observation，因此不会扩大 rollout 中的 reference target。
原有 SPV5 reference encoder、height/contact estimator 及其监督损失保持
不变。

## 梯度与更新契约

PMOE 有三条相互分离的更新路径：

1. PPO 策略损失更新 shared policy、8 个 experts、action head 和动作分布。
2. PAE 仅由窗口重建 MSE 更新，使用独立 optimizer、learning rate 和
   gradient clipping。
3. K-Means prototypes 是 buffers，没有 trainable parameters；每次 PPO
   update 开始时，对刚收集的 rollout 做一次 no-grad EMA 更新。

MoE 在最内层再次对 route probability 执行 `detach`。即使上层误传入
带计算图的 route，PPO 梯度也无法到达 PAE。PAE 参数虽然为兼容现有
checkpoint 结构仍出现在 actor optimizer 的参数组中，但独立 PAE
optimizer step 后会清空其梯度，主 actor optimizer 不会对它执行第二次
更新。

行为采样时的 route 会写入 rollout cache。后续多个 PPO epoch 始终复用
行为时的 detached route，避免 PAE 或 prototypes 在 update 期间变化后
改变同一批数据的专家身份。第一个 rollout 尚无 prototypes，使用均匀
route；第一次 PPO update 后完成初始化。

多 GPU 训练中，特征统计、cluster sums 和 cluster counts 都先做全局
all-reduce；prototype 初始化由 rank 0 完成后广播。

## 默认配置

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `pmoe_num_experts` | 8 | prototype 与 expert 数量 |
| `pmoe_top_k` | 2 | 每个窗口激活的 experts |
| `pmoe_pae_latent_dim` | 8 | PAE 周期潜在通道数 |
| `pmoe_pae_hidden_dims` | `[64, 64]` | PAE Conv1d 隐层 |
| `pmoe_cluster_temperature` | 1.0 | 距离转 soft route 的温度 |
| `pmoe_cluster_momentum` | 0.99 | 特征统计和 prototype 的 EMA 动量 |
| `pmoe_pae_learning_rate` | `5e-5` | 独立 PAE optimizer 学习率 |
| `pmoe_pae_loss_coef` | 1.0 | PAE 重建损失权重 |

## 训练诊断

PAE 指标：

- `pmoe_pae_mse`：完整规范化窗口重建 MSE；
- `pmoe_pae_root_mse`、`pmoe_pae_joint_mse`：root 与 joints 分项误差；
- `pmoe_pae_mean_amplitude`：周期通道平均幅值；
- `pmoe_pae_active_channel_fraction`：幅值大于 `1e-3` 的通道比例。

聚类指标：

- `pmoe_cluster_effective_count`：由 cluster usage entropy 计算的有效簇数；
- `pmoe_cluster_usage_min/max`：最少/最多使用簇的比例；
- `pmoe_cluster_empty_count`：当前 rollout 未分配样本的簇数；
- `pmoe_cluster_mean_distance`：样本到最近 prototype 的平均标准化距离；
- `pmoe_cluster_update_count`：prototype 更新次数。

PAE 是否形成了对路由有效的表示，不能只由重建 MSE 判断。至少应同时
满足：重建误差下降、活动通道未坍塌、有效簇数不长期接近 1、空簇数量
可控，并在与 SPV5-2 的同数据同 seed 对照中改善 tracking 指标。
