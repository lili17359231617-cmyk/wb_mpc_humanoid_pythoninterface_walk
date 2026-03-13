# G1 RL 调节 MPC 权重 — 训练记录书

本文档用于记录「RL 调节 MPC 权重」训练方式、脚本关系与每次训练结果，便于复现与迭代。阶段规划见 [RL_MPC_权重训练_阶段性计划.md](./RL_MPC_权重训练_阶段性计划.md)

---

## 一、脚本与包内文件关系

### 1.1 入口脚本

| 脚本 | 用途 |
|------|------|
| **`scripts/train_g1_ppo_v1.py`** | PPO 训练入口：单 env、G1MpcWeightEnv，支持从头训练与从 checkpoint 续训。 |
| **`scripts/test_mpc_rl.py`** | 环境测试入口：用随机策略跑少量步，验证环境与 C++ 扩展是否正常、观测/动作维度与 reward 是否合理。 |

两者均需在导入 `humanoid_wb_mpc` 前将 C++ 扩展路径加入 `sys.path`，并可选设置 `MUJOCO_GL`（无头用 `egl`，有显示用 `glfw`）。

### 1.2 `test_mpc_rl.py` 与包内子文件的关系

- **`humanoid_wb_mpc` 包**（`humanoid_wb_mpc/`）
  - **`bootstrap`**：在导入包时执行，配置 C++ 扩展路径与 MuJoCo 渲染后端，`test_mpc_rl.py` 在导入包前设置 `MUJOCO_GL`，随后 `import humanoid_wb_mpc.bootstrap`。
  - **`make_g1_mpc_weight_env`**（由 `__init__.py` 从 `envs` 暴露）：构造「仿真同步、多速率」的 RL 环境；`test_mpc_rl.py` 直接调用 `make_g1_mpc_weight_env(headless=..., seed=42)` 得到环境。
  - **`envs/G1MpcWeightEnv.py`**：实现 `G1MpcWeightEnv`（Gymnasium 兼容）、`make_g1_mpc_weight_env`；提供 `reset`/`step`、观测/动作空间、`max_episode_steps`（time limit）等。
  - **`config/`**：提供 `DEFAULT_TASK_FILE`、`DEFAULT_URDF_FILE`、`DEFAULT_REF_FILE`、`DEFAULT_RL_FREQUENCY_HZ` 等；环境与训练脚本均通过 `humanoid_wb_mpc.config` 使用这些默认路径与频率。
  - **`utils/reward_handler.py`**：`MpcWeightEnvReward` 等奖励逻辑，被 `G1MpcWeightEnv` 在 `step()` 中调用。
  - **`spaces.py`**：`MpcResidualActionSpace`（动作裁剪等），被 `G1MpcWeightEnv` 用于动作空间与 step 内裁剪。

关系小结：**`test_mpc_rl.py`** 只依赖「包根 + bootstrap + make_g1_mpc_weight_env」；后者内部依赖 **config、envs/G1MpcWeightEnv、spaces、utils/reward_handler**，以及 C++ 扩展（通过 bootstrap 与安装路径下的 `humanoid_wb_mpc_py`）。

### 1.3 训练脚本与包内子文件的关系

- **`train_g1_ppo_v1.py`** 使用与 `test_mpc_rl.py` 相同的包结构，另外：
  - 使用 **Stable-Baselines3 (PPO)**，通过 `make_g1_mpc_weight_env(...)` 得到环境后交给 PPO；可选 `RESUME_FROM` 加载已有 `.zip` 继续训练。
  - TensorBoard 日志目录由 `TB_LOG_DIR` 或默认 `./ppo_g1_logs/phase1_short` 决定；训练曲线含 `rollout/ep_rew_mean`、`rollout/ep_len_mean`（需环境有 time limit 才会每 rollout 有统计）。

---

## 二、当前训练方式

- **环境**：`G1MpcWeightEnv`，单 env，仿真同步多速率（RL 50Hz : MPC 100Hz : Sim 1000Hz），`max_episode_steps=2048`（默认），倒地或达到步数即结束 episode。
- **算法**：PPO（SB3），`n_steps=2048`，`batch_size=128`，`learning_rate=1e-4`，其余超参见 `train_g1_ppo_v1.py`。
- **日志**：TensorBoard 写入 `TB_LOG_DIR`（默认 `./ppo_g1_logs/phase1_short`）；查看曲线建议用绝对路径或先 `cd` 到脚本目录。
- **从头训练**：`TOTAL_TIMESTEPS` 表示「总训练步数」；保存名为 `g1_ppo_residual_v1_phase1_{TOTAL_TIMESTEPS}.zip`。
- **从 checkpoint 续训**：设置 `LOAD_MODEL` 为已有 `.zip` 路径（见下文「复现 .zip 模型」），`TOTAL_TIMESTEPS` 表示本次训练的总步数；保存名按当前 phase 与步数重新命名。
- **reset 随机化（站立鲁棒性）**：`G1MpcWeightEnv` 在 `reset` 中可选对「初始高度/姿态、velocity_command、短脉冲外力、地面摩擦系数」做小范围随机（默认开启）。构造环境时 `enable_reset_randomization=True`（默认）；设为 `False` 可关闭随机化，便于复现或调试。

### 2.1 复现 `.zip` 模型（如 `ppo_zip/phase1/g1_ppo_residual_v1_phase1_500000_n4.zip`）

该类文件是 Stable-Baselines3 保存的 PPO 检查点，有两种复现方式：

**方式一：续训（继续训练）**
从该检查点接着训练（例如进入 Phase 2，或同 Phase 再训更多步）。在 `scripts` 目录下：

```bash
cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc/humanoid_nmpc/humanoid_wb_mpc/python/scripts

# 示例：从 Phase 1 的 50 万步模型继续训 Phase 2
LOAD_MODEL=ppo_zip/phase1/g1_ppo_residual_v1_phase1_500000_n4.zip \
PHASE=2 N_ENVS=4 HEADLESS=1 TOTAL_TIMESTEPS=300000 \
TB_LOG_DIR=./ppo_g1_logs/phase2 \
python3 train_g1_ppo_v1.py
```

- `LOAD_MODEL`：指向已有 `.zip` 的路径（相对脚本目录或绝对路径均可）。
- 若文件存在，会加载该模型并沿用当前 `PHASE`、`N_ENVS` 等配置继续 `learn`；保存名按本次 `phase` 与 `total_steps` 重新生成（如 `g1_ppo_residual_v1_phase2_300000_n4.zip`）。

**方式二：评估/回放（只看策略表现，不训练）**
使用 `test_mpc_rl.py`，通过环境变量 `MODEL` 指定 `.zip` 路径即可用该策略回放（脚本会加载 PPO 并用 `model.predict` 替代随机动作）：

```bash
cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc/humanoid_nmpc/humanoid_wb_mpc/python/scripts

# 无头回放（默认 2048 步，可设 EVAL_STEPS=500 等）
MODEL=ppo_zip/phase1/g1_ppo_residual_v1_phase1_500000_n4.zip HEADLESS=1 python3 test_mpc_rl.py

# 有显示时开 MuJoCo 窗口看机器人
MODEL=ppo_zip/phase1/g1_ppo_residual_v1_phase1_500000_n4.zip HEADLESS=0 python3 test_mpc_rl.py
```

- `MODEL`：指向 `.zip` 的路径（相对脚本目录或绝对路径）。
- `EVAL_STEPS`：回放总步数，默认 2048；步数内若 episode 终止会自动 `reset` 继续。
- `DETERMINISTIC`：默认 1，即确定性策略；设为 0 可带随机性。

### 2.2 从 20 万步 checkpoint 再训（续训命令示例）

在 `scripts` 目录下，模型与脚本同目录时：

```bash
LOAD_MODEL=g1_ppo_residual_v1_phase1_200000.zip \
TOTAL_TIMESTEPS=20000 PHASE=1 \
python3 train_g1_ppo_v1.py
```

- 若模型在其他路径，将 `LOAD_MODEL` 改为该 `.zip` 的绝对或相对路径即可。

### 2.3 TensorBoard 曲线

```bash
tensorboard --logdir /wb_humanoid_mpc_ws/src/wb_humanoid_mpc/humanoid_nmpc/humanoid_wb_mpc/python/scripts/ppo_g1_logs/phase1_short

- 若模型在其他路径，将 `RESUME_FROM` 改为该 `.zip` 的绝对或相对路径即可。
- 续训完成后新模型保存为：`g1_ppo_residual_v1_phase1_200000_plus_20000.zip`。

---

## 三、训练结果记录（填写用）

0.1、代码逻辑验证阶段 phase0:

| 从头 | 100000 |  曲线图见phase0_short/PPO_1(已删除)  | g1_ppo_residual_v1_phase1_100000.zip（已删除） |
| 从头 | 200000 |  曲线图见phase0_short/PPO_2  | g1_ppo_residual_v1_phase1_100000.zip（已删除） |
| 从头 | 200000 |  曲线图见phase0_short/PPO_4  | g1_ppo_residual_v1_phase1_200000.zip（已删除） |
| 从头（n=4) | 200000 | 曲线图见phase0_short/PPO_6 | g1_ppo_residual_v1_phase1_200000.zip（已删除）  |
|------|------|-------------|----------------------------------------|---------------------------|---------------------------|--------------|
结论：mujoco视图中机器人稳定站立；已经跑了 1e5～2e5 步，多次实验，并看了 TensorBoard；曲线诊断：rollout/ep_len_mean ≈ 2048 打满 time limit，ep_rew_mean 明显上升，说明“能学得动”、且基本不早倒。”说明已经打通闭环：能稳站、能学习。
但最后的并行训练结果曲线图很奇怪，需要检查原因

0.1*、 phase0 debug:
意外情况：给环境加随机扰动代码和检查并行训练逻辑后，训练结束变得很快，机器人出现不合理行为：在可视化窗口中表现为，一开始训练就穿模+起飞

debug:经过各类调试和代码检查，发现代码逻辑没有问题
解决方法：
1）重新配置了一遍容器环境和依赖；
2）Q矩阵权重累乘漂移：在模块构造时保存 Q_base_，每次更新时从基准出发
3）奖励函数轴向赋值错误：奖励函数把 yaw（偏航）当作 roll（横滚）来惩罚，真正的横滚方向 obs[5] 完全未被监督，已修改正确
4）奖励速度目标与指令不一致：存储并每步传递当前速度指令
5）动作空间中3个维度无效（初始值为0）：公式从 Q_base * exp(a) 改为 Q_base * (1 + exp(a))
6）RobustnessMetricsCallback 多env状态污染：首次出现新索引时懒初始化
7）观测空间增强：58+2+1+58，添加了接触标志、步态相位变量、上一步 action
8）奖励函数设计（全为负数）：添加存活奖励 ，减小跌倒惩罚量级

**机器人出现不合理行为主要原因：容器中的环境和依赖乱了，numpy在同一个容器中安装了两个不同的版本，导致了环境紊乱

训练验证（加入少量随机扰动）
| 从头 | 200000 | 曲线图见phase0_afterdebug/PPO_1  | g1_ppo_phase0_afterdebug1_200000.zip |

结论：训练恢复正常，可以按阶段正式开始训练迭代

---

1.1  V1
初阶方案1：两阶段分离，按顺序训练

1.1.1
Phase 1：站立鲁棒性
具体说明：
  - 速度指令 = 0（纯站立）
  - 强domain rand：外力概率提高到70%，力度30~80N
  - 只调整与基座相关的Q维（indices 2,4,5,29-34）
  - 成功标准：fall_rate_ema < 0.1

Phase 1操作：
1）暂时不改动超参（n_steps=2048, batch_size=128, lr=1e-4, ent_coef=0.01），在看到如下情况时，再回过头针对性调参：
      reward 明显还能上升但进展很慢；曲线抖得很厉害、时好时坏；换成并行 env 后不稳定。
2）在环境中注入随机物理干扰，完成站立任务下的鲁棒性测试（外力扰动、小随机化）。
**Phase 1 实际生效的扰动**（`G1MpcWeightEnv.reset` + `train_g1_ppo_v1.py` Phase 1 配置）如下：
  - **初始基座位置**：xy ±0.012 m，z ±0.02 m。
  - **初始基座姿态（欧拉角）**：各轴 ±0.02 rad。
  - **初始关节角**：各关节 ±0.012 rad。
  - **地面摩擦系数**：μ ∈ [0.65, 1.08]，每 episode 采样一次。
  - **速度指令**：Phase 1 为纯站立，`vel_cmd_rand_range=(0,0,0)`，不做速度随机化。
  - **外力脉冲**：以 **70% 概率**（`FORCE_PROB`，默认 0.70）在 pelvis 上施加一次脉冲；水平方向随机（0~2π），大小由 `force_mag_range`（Phase 1 默认 30~80 N），竖向 fz ∈ [-8, 12] N，持续 12~22 个仿真步。可通过环境变量 `FORCE_PROB`、`FORCE_MAG_MIN`/`FORCE_MAG_MAX` 调整。
3）训练
| 从头 | 500000 | n=4, headless | 曲线图见phase1/PPO_1 | g1_ppo_residual_v1_phase1_500000_n4.zip |
**结论**：无跌倒（terminated/fall_rate_ema=0），base_drift 小且平稳。
损失下降、value 与 explained_variance 改善，策略与价值在稳步学习。
KL、clip_fraction、entropy、std 均处于合理范围，训练稳定

| n4,续训 | 200000 | 从 50 万步 n4 续训 | TB 见 phase1_续训/PPO_1 | g1_ppo_residual_v1_phase1_续200000_n4.zip |
**结论**：续训后 train/explained_variance 由 50 万步末尾约 -0.07～-0.09 恶化到 -0.24～-0.33；value_loss 降至约 120、loss 约 57。说明同条件继续加步数收益有限，价值估计与回报匹配变差。**Phase 1 正式产出采用 50 万步 n4 模型，已进入 Phase 2。**

---

1.1.2
Phase 2：行走稳定性
具体说明（见 `train_g1_ppo_v1.py`）：
  - 速度指令：基础 vx=0.15 m/s，vy=0、高度 0.75、yaw_rate=0；随机化范围 vel_rand=(0.30, 0.15, 0.15)。
  - 奖励：w_velocity=1.0、w_gait_symmetry=0.2，其余与 Phase 1 类似。
  - 外力：FORCE_PROB=0.45，力度 28~62 N。
  - 调节维度：PHASE2_ACTIVE_DIMS（base + 腿部 Q 维 6–17）。

**启动命令（从 Phase 1 的 50 万步模型续训）**：
**机器 A（Phase 2 行走导向）**：
```bash
cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc/humanoid_nmpc/humanoid_wb_mpc/python/scripts

LOAD_MODEL=ppo_zip/phase1/g1_ppo_residual_v1_phase1_500000_n4.zip \
PHASE=2 N_ENVS=4 HEADLESS=1 TOTAL_TIMESTEPS=300000 \
TB_LOG_DIR=./ppo_g1_logs/phase2 \
LOG_ROBUST_METRICS=1 \
python3 train_g1_ppo_v1_行走导向.py
```
查看曲线：`tensorboard --logdir ./ppo_g1_logs/phase2`
```

4）训练1
| n4,续训 | 300000 | 从 50 万步 n4 续训 | TB 见 phase2/PPO_1 | 20%后某一个并行 env 里 MPC 的 QP 求解失败，导致该子进程退出，未生成 |
| n4,续训 | 300000 | 从 50 万步 n4 续训 | TB 见 phase2/PPO_2 | 27%后某一个并行 env 里 MPC 的 QP 求解失败，导致该子进程退出，未生成 |
日志显示动力学违反爆炸 → 状态/轨迹超界

对应处理：
a、捕获 MPC 崩溃为正常 episode 终止, 返回 terminated=True，reward=-10（大惩罚）,子进程继续存活，环境自动 reset 进入下一 episode;
b、修改终止高度: 0.35m → 0.6m;
c、暂时缩小 action clip 范围： [-1.0, 1.0] → [-0.4, 0.4] 来防止早期训练阶段的过激调整。修改两处代码，spaces.py；G1MpcWeightEnv.py，self.action_space = spaces.Box；
d、Phase 1 模型训练时 action space 是 Box(-1.0, 1.0, (58,))：加载时用 custom_objects 跳过 space 检查

5）训练2
| n4,续训 | 300000 | 从 50 万步 n4 续训，action clip [-0.4, 0.4] | TB 见 phase2/PPO_3 | g1_ppo_residual_v1_phase2_300000_n4.zip |
**Phase 2 PPO_3 曲线分析**（step 约 32k～328k，共 10 个记录点）：
- **鲁棒性**：`robust/fall_rate_ema` 均值约 0.36、范围 0.25～0.47，末点 0.32。跌倒率明显高于 Phase 1 目标（<0.1），行走任务下仍有较多 episode 因倒地结束。`robust/base_drift_xy_m` 均值约 33.9 m、末点 46.7 m（行走时随速度与时长累积，属正常量级）。`robust/terminated` 记录为 0（与 callback 上报方式有关）。
- **训练指标**：`train/explained_variance` 由首点 0.68 升至末点 **0.93**，价值拟合良好；`train/loss` 约 32→18、`train/value_loss` 约 77→43，整体下降；`train/approx_kl` 约 0.008～0.011，`train/clip_fraction` 约 0.30～0.41，步长与裁剪正常。
- **结论**：在缩小 action clip 并做 MPC 崩溃捕获后，本 run **完整跑完** 30 万步未中途崩进程。策略在行走任务上仍有约 25%～47% 的跌倒率 EMA，需后续通过更长训练、奖励微调或更保守速度指令进一步压低；价值与 loss 趋势健康，可在此基础上继续迭代或保存为 Phase 2 初版 checkpoint。


1.2
初阶方案2：在Phase 1基础上进一步加强抗干扰能力（phase1.5)

1.2.1
Phase 1：站立鲁棒性（全部同上
具体说明：
  - 速度指令 = 0（纯站立）
  - 强domain rand：外力概率提高到70%，力度30~80N
  - 只调整与基座相关的Q维（indices 2,4,5,29-34）
  - 成功标准：fall_rate_ema < 0.1

......................
......................

1）训练
| 从头 | 500000 | n=4, headless | 曲线图见phase1/PPO_1 | g1_ppo_residual_v1_phase1_500000_n4.zip |
**结论**：无跌倒（terminated/fall_rate_ema=0），base_drift 小且平稳。
损失下降、value 与 explained_variance 改善，策略与价值在稳步学习。
KL、clip_fraction、entropy、std 均处于合理范围，训练稳定

2）| 续训 | 200000 | 从 50 万步 n4 续训，TB 见 phase1_续训/PPO_1 | g1_ppo_residual_v1_phase1_续200000_n4.zip |
**结论**：续训后 train/explained_variance 由 50 万步末尾约 -0.07～-0.09 恶化到 -0.24～-0.33；value_loss 降至约 120、loss 约 57。说明同条件继续加步数收益有限，价值估计与回报匹配变差。**Phase 1 正式产出采用 50 万步 n4 模型，进入 Phase 2“**

1.2.2
Phase 1.5_heavy_disturb：加大扰动
具体说明（见 `train_g1_ppo_v2_站立抗扰动导向.py`）：
  - 速度指令：纯站立 vx=0、vy=0、高度 0.75、yaw_rate=0；随机化 vel_rand=(0, 0, 0)。
  - 奖励：w_height=1.5、w_orientation=0.8、w_velocity=0，其余与 Phase 1 类似。
  - 外力：FORCE_PROB=0.85，力度 45~100 N。
  - 调节维度：STANDING_ACTIVE_DIMS（base 相关：高度、roll/pitch、base 线/角速度 [2,4,5,29–34]）。

3）训练
| 续训n=6 | 300000 | 从 50 万步 n4 续训 | TB 见  phase1.5_heavy_disturb/PPO_1 | phase1.5_heavy_disturb/g1_ppo_standing_heavy_disturb_300000.zip |
分析：
a、几乎 0 跌倒、满长度 episode、较小 base drift:策略已经“扛得住”，属于鲁棒性比较强但略偏保守的站立控制器。
b、explained_variance 持续变负、ep_rew_mean 稍降：提示继续在同一 reward / 同一扰动范围上再刷步数，可能只是把 value 越学越差，对策略本身收益不大。

**结论（Phase 2 重扰动结果总结 + 是否作为正式模型采用）**：
本次 30 万步续训在「加大扰动」（FORCE_PROB=0.85、45～100 N）下达到几乎 0 跌倒、满长 episode、base_drift 约 2～3 cm，站立抗扰动目标已达成。**建议将本阶段产出作为 Phase 2 站立抗扰动正式模型采用**，推荐 checkpoint：`ppo_zip/phase1.5_heavy_disturb/g1_ppo_standing_heavy_disturb_300000.zip`。不再建议在同一设定下继续加步数；后续可转向「轻行走 + 扰动」或调整 reward 做短训，或先用 `test_mpc_rl.py` 做定性推扰验证后再进入下一阶段。

1.2.3
Phase 2：行走稳定性
  - 速度指令：基础 vx=0.15 m/s，vy=0、高度 0.75、yaw_rate=0；随机化范围 vel_rand=(0.30, 0.15, 0.15)。
  - 奖励：w_velocity=1.0、w_gait_symmetry=0.2，其余与 Phase 1 类似。
  - 外力：FORCE_PROB=0.45，力度 28~62 N。
  - 调节维度：PHASE2_ACTIVE_DIMS（base + 腿部 Q 维 6–17）。

4）训练
| 续训n=6 | 300000 | 从 g1_ppo_standing_heavy_disturb_300000.zip 续训 | TB 见  phase2_heavy_disturb_续训/PPO_1 | phase2/g1_ppo_standing_heavy_disturb_300000 |
**结论**
抗跌倒：续训后仍保持 0 跌倒、满长 episode，可认为站立抗扰动（不摔）目标仍然满足。
基座漂移：续训使 base_drift 明显增大（约 2 cm → 5～6 cm），若你关心「站立时少滑移」，这段续训是不利的。
价值与回报：explained_variance 继续变负、ep_rew_mean 略降，说明同设定续训对策略与价值估计的边际收益有限，甚至略有负面。

V1 总结论：
下一步行走相关的续训 / 加扰动训练，建议以 phase2/PPO_3 对应的 checkpoint 作为基准。
对 heavy_disturb 续训 run，更适合作为「站立抗扰动」方向的一个参考结果，而不是行走训练主线的初始化模型。


2.1 V2 （同逻辑从头重新训练）
操作：
1）修改 “权重更新公式“ ：
发现之前进行训练采用的权重更新公式是Q_new=Q_base * (1 + exp(a))，这样的Q_new永远大于Q_base，且当Q_base=0时更新无效
更改权重更新逻辑为如下，权重初始值为0的使用Softplus 映射，即Qnew​=α⋅ln(1+exp(a+b)）；初始值不为0的采用指数残差Qnew​=Qbase​⋅exp(a)
其中，初始α=0.1,b=-2, a ∈ [-1, 1]
2）奖励函数修改：
原本的奖励函数多采用-e^2，容易导致在快摔倒时过度惩罚，阅读论文了解到“高斯核函数”

| 奖励组成         | 修改前逻辑                                                                 | 修改后逻辑                                                                                           | 说明                             |
|------------------|----------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|----------------------------------|
| 速度/指令跟踪    | L1 误差：\(-w_\text{vel} (|v_x - v_{x}^\ast| + |v_y - v_{y}^\ast|)\)      | 高斯核：\(\exp(-v_\text{err}^2 / \sigma_v) + \exp(-\text{yaw\_err}^2 / \sigma_\text{yaw})\)         | 减少大误差时的过度惩罚           |
| 姿态稳定         | L1 误差：\(-w_\text{ori}(\min(|roll|, r_{\max}) + \min(|pitch|, r_{\max}))\) | 高斯核：\(\exp(-(roll^2 + pitch^2)/\sigma_\text{stable})\)                                          | 保持直立姿态，同时更平滑         |
| 权重幅值惩罚     | \(-w_\text{action\_mag} \sum a_t^2\)                                      | 已移除（由 Q 平滑隐含约束）                                                                          | 避免直接压制动作幅值             |
| 权重平滑惩罚     | \(-w_\text{action\_smooth} \sum (a_t - a_{t-1})^2\)                       | \(-0.05 \sum (Q_{t} - Q_{t-1})^2\)，其中 \(Q_t\) 为真实 Q_new 对角线                               | 直接约束 MPC 权重变化的平滑性   |
| 生存奖励/跌倒惩罚 | 常数生存奖励 \(w_\text{survival}\) + 跌倒罚分 \(fall\_penalty\)           | 生存奖金：未跌倒时 \(r_\text{alive}=1.0\)，否则 0                                                   | 仍鼓励长 episode、不跌倒         |
| 总奖励形式       | 各分量线性加权求和                                                         | \(R = w_1 r_\text{tracking} + w_2 r_\text{stable} + w_3 r_\text{smooth} + r_\text{alive}\)          | 结构更清晰，便于调节各子目标权重 |








## 四、与阶段性计划的对应

- 本记录书中的「当前训练方式」与「续训命令」对应阶段性计划中**第一阶段**的「短训练试跑」「得到 reward / episode length 曲线」及「可复现」要求。
- 训练入口为 `scripts/train_g1_ppo_v1.py`，环境测试入口为 `scripts/test_mpc_rl.py`，与计划文档中的「与现有文档、代码的对应关系」一致。
- 后续若增加 4～8 并行 env、调整奖励/终止或 PPO 超参，可在本记录书「当前训练方式」与「训练结果记录」中增补说明与行记录。

---

*文档版本：1.0 | 与 train_g1_ppo_v1.py、test_mpc_rl.py 及 humanoid_wb_mpc 包结构对应*
