# G1 RL 调节 MPC 权重 — 训练记录书

本文档用于记录「RL 调节 MPC 权重」训练方式、脚本关系与每次训练结果，便于复现与迭代。阶段规划见 [RL_MPC_权重训练_阶段性计划.md](./RL_MPC_权重训练_阶段性计划.md)（若该文件为空，可参考同目录下 `RL_MPC_权重训练记录.md` 中的阶段性计划正文）。

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
- **从 checkpoint 续训**：设置 `RESUME_FROM` 为已有 `.zip` 路径，`TOTAL_TIMESTEPS` 表示「追加步数」，时间轴不重置；保存名为 `{原模型名}_plus_{追加步数}.zip`。
- **reset 随机化（站立鲁棒性）**：`G1MpcWeightEnv` 在 `reset` 中可选对「初始高度/姿态、velocity_command、短脉冲外力、地面摩擦系数」做小范围随机（默认开启）。构造环境时 `enable_reset_randomization=True`（默认）；设为 `False` 可关闭随机化，便于复现或调试。

### 2.1 从 20 万步 checkpoint 再训 2 万步（续训命令）

在 `scripts` 目录下执行（模型文件与脚本同目录且名为 `g1_ppo_residual_v1_phase1_200000.zip` 时）：

```bash
cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc/humanoid_nmpc/humanoid_wb_mpc/python/scripts

TOTAL_TIMESTEPS=20000 \
RESUME_FROM=g1_ppo_residual_v1_phase1_200000.zip \
python3 train_g1_ppo_v1.py
```
### 2.2 tensorboard 曲线显示命令：
tensorboard --logdir /wb_humanoid_mpc_ws/src/wb_humanoid_mpc/humanoid_nmpc/humanoid_wb_mpc/python/scripts/ppo_g1_logs/phase1_short

- 若模型在其他路径，将 `RESUME_FROM` 改为该 `.zip` 的绝对或相对路径即可。
- 续训完成后新模型保存为：`g1_ppo_residual_v1_phase1_200000_plus_20000.zip`。

---

## 三、训练结果记录（填写用）

1、

| 日期 | 类型 | 总/追加步数 | 备注（如 seed、headless、RESUME_FROM） | rollout/ep_rew_mean（约） | rollout/ep_len_mean（约） | 保存的模型名 |
|------|------|-------------|----------------------------------------|---------------------------|---------------------------|--------------|
| 20260302 | 从头 | 100000 | seed=42, headless | 无 | 2048（打满 time limit） | g1_ppo_residual_v1_phase1_100000.zip |
| 20260302 | 从头 | 200000 | seed=42，headless | 无 | 2048 （打满 time limit）| g1_ppo_residual_v1_phase1_200000.zip |
| 20260302 | 续训 | 为20000追加100000 | seed=42，head | -3149->-3102上升 | 2048 （打满 time limit）| g1_ppo_residual_v1_phase1_200000.zip |
|------|------|-------------|----------------------------------------|---------------------------|---------------------------|--------------|
结论：mujoco视图中机器人稳定站立；已经跑了 1e5～2e5 步，多次实验，并看了 TensorBoard；曲线诊断：rollout/ep_len_mean ≈ 2048 打满 time limit，ep_rew_mean 明显上升，说明“能学得动”、且基本不早倒。
第一阶段的“站立基础版”已经打通闭环：能稳站、能学习。

说明：

- **类型**：填「从头」或「续训」。
- **总/追加步数**：从头训练填总步数；续训填本次追加步数（如 20000）。
- **rollout/ep_rew_mean**、**rollout/ep_len_mean**：来自 TensorBoard 或训练结束时终端打印的 `rollout/` 行；ep_len_mean 长期为 2048 表示 episode 多因 time limit 结束、少因倒地结束。
- 若某次仅为短时试跑或中断，可在备注中说明，并在曲线列填「未记」或「中断」。

---

2、
操作：
1）暂时不改动超参（n_steps=2048, batch_size=128, lr=1e-4, ent_coef=0.01），在看到如下情况时，再回过头针对性调参：reward 明显还能上升但进展很慢；曲线抖得很厉害、时好时坏；换成并行 env 后不稳定。
2）在环境中注入随机物理干扰，完成站立任务下的鲁棒性测试（外力扰动、小随机化）

## 四、与阶段性计划的对应

- 本记录书中的「当前训练方式」与「续训命令」对应阶段性计划中**第一阶段**的「短训练试跑」「得到 reward / episode length 曲线」及「可复现」要求。
- 训练入口为 `scripts/train_g1_ppo_v1.py`，环境测试入口为 `scripts/test_mpc_rl.py`，与计划文档中的「与现有文档、代码的对应关系」一致。
- 后续若增加 4～8 并行 env、调整奖励/终止或 PPO 超参，可在本记录书「当前训练方式」与「训练结果记录」中增补说明与行记录。

---

*文档版本：1.0 | 与 train_g1_ppo_v1.py、test_mpc_rl.py 及 humanoid_wb_mpc 包结构对应*
