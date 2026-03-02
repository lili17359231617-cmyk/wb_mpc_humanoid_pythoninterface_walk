# G1 人形机器人「RL 调节 MPC 权重」强化学习训练说明书

本文档说明如何使用 `test_mpc_RL.py` 提供的 **G1MpcWeightEnv** 环境进行强化学习训练，实现「RL 调节 MPC 权重」的架构。

---

## 1. 概述与架构

### 1.1 设计目标

- **控制架构**：MPC 负责全身轨迹与接触力优化，RL 不直接输出关节指令，而是**输出 MPC 代价函数中状态/输入权重的残差调整**，通过 `MpcWeightAdjustmentModule` 映射为
  `Q_i = Q_base_i * exp(a_i)`，从而在保持 MPC 稳定性的前提下，让 RL 学习在不同工况下微调权重以提升表现。
- **仿真方式**：**仿真同步、多速率分层**——不启动 MPC 后台线程，由仿真时间驱动。多速率组合为 **RL(50Hz) : MPC(100Hz) : Sim(1000Hz)**：
  - **RL 50 Hz**：每 0.02 s 输出一次 58 维残差权重，并在此周期内推进仿真；
  - **MPC 100 Hz**：每个 RL 步内执行 2 次 MPC 求解与关节控制；
  - **Sim 1000 Hz**：每次 MPC 后执行 10 次仿真步（sim_dt=0.001 s），即每 RL 步共 20 次仿真步。

### 1.2 与实时测试脚本的区别

| 项目         | 原 test_mpc_mrt_walk（实时异步）     | test_mpc_RL（RL 训练环境）           |
|--------------|--------------------------------------|--------------------------------------|
| MPC 执行     | 后台线程异步、按固定频率               | 当前线程同步、每 RL 步内 2 次 MPC     |
| 仿真步长     | 控制周期相关                          | Sim 1000 Hz（0.001 s/步）            |
| 时间驱动     | 墙钟时间 + 仿真时间                   | 仅仿真时间                           |
| RL 接口      | 无                                    | Gymnasium 风格 reset/step             |
| 动作         | 无                                    | 58 维残差权重 → MpcWeightAdjustmentModule |

---

## 2. 环境要求与安装

### 2.1 依赖

- **工作空间**：`wb_humanoid_mpc_ws` 已成功编译，并已安装 `humanoid_wb_mpc` 的 Python 绑定（生成 `humanoid_wb_mpc_py`）。
- **Python**：建议 3.8+，需 `numpy`、`scipy`（用于欧拉/四元数转换）。
- **可选**：若使用 Stable-Baselines3 / Gymnasium 等库训练，需另行安装对应包。

### 2.2 确认 Python 绑定可用

```bash
cd /wb_humanoid_mpc_ws
source install/setup.bash  # 若使用 ROS2 工作空间
# 将安装目录下的 lib 加入 PYTHONPATH（脚本内已写默认路径）
python3 -c "import humanoid_wb_mpc_py as mpc_py; print(mpc_py.WBMpcInterface)"
```

若无报错，说明可正常导入。

### 2.3 无头/有头运行

- **无头（服务器/批量训练）**：`HEADLESS=1` 或构造环境时 `headless=True`，使用 EGL 渲染。
- **有头（本地调试、看步态）**：`HEADLESS=0` 且具备 DISPLAY，使用 GLFW。

### 2.4 重新编译

修改 C++ 代码（如 `python_binding.cpp`）或 MPC 相关源文件后，需要重新编译并 `source` 安装目录后再运行 Python 脚本。

**仅重编 humanoid_wb_mpc 包（改过 Python 绑定或本包 C++ 时）：**

```bash
cd /wb_humanoid_mpc_ws
source /opt/ros/jazzy/setup.bash   # 若已 source 可省略
source install/setup.bash          # 已有 install 时建议先 source

colcon build --packages-select humanoid_wb_mpc --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

source install/setup.bash
```

或用工作空间内 Makefile：

```bash
cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc
make build PKG=humanoid_wb_mpc
cd /wb_humanoid_mpc_ws && source install/setup.bash
```

**全工作空间重编：**

```bash
cd /wb_humanoid_mpc_ws
source /opt/ros/jazzy/setup.bash

colcon build --parallel-workers 2 --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

source install/setup.bash
```

或：

```bash
cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc
make build-all
cd /wb_humanoid_mpc_ws && source install/setup.bash
```

**先清再编（该包完全重编）：**

```bash
cd /wb_humanoid_mpc_ws
rm -rf build/humanoid_wb_mpc install/humanoid_wb_mpc
colcon build --packages-select humanoid_wb_mpc --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

---

## 3. 动作空间、观测空间与奖励

### 3.1 动作空间（Action）

- **类型**：连续，**58 维**向量（与 MPC 状态/Q 维一致）。
- **含义**：RL 策略每步输出 58 维向量 `a`，通过 `MpcWeightAdjustmentModule.set_residual_weights(a)` 注入；在 C++ 侧用于调整 MPC 状态代价矩阵 Q：`Q_i = Q_base_i * exp(a_i)`。
- **范围**：默认 `[-1, 1]`（由 `MpcResidualActionSpace` 的 `low/high` 决定），一般无需再放大，因 `exp(a)` 已能带来明显缩放。
- **使用注意**：
  - 与 Gymnasium 的 `Box` 不同，本环境使用自定义 `MpcResidualActionSpace`，需在封装时用 `env.action_space.low/high` 或 `env.action_space.clip(action)` 做裁剪。
  - 若用 Stable-Baselines3，建议用 `VecNormalize` 或自定义 `clip` 将策略输出裁到 `[-1,1]`。

### 3.2 观测空间（Observation）

- **类型**：连续向量，维度 = **58**（与 `interface.get_state_dim()` 一致，随模型配置可能变化）。
- **布局**（OCS2 MPC 状态）：
  `[base_pos(3), base_euler(3), joint_pos(23), base_vel(3), euler_dot(3), joint_vel(23)]`
  即：基座位置、基座欧拉角、关节位置、基座线速度、欧拉角导数、关节速度。
- **来源**：每步由当前仿真 `RobotState` 经 `controller.get_mpc_state_from_robot_state(robot_state)` 得到，与 MPC 内部状态一致。
- **归一化**：环境不做自动归一化；若训练不稳定，可在外部对 obs 做标准化或归一化（如按历史统计或固定范围）。

### 3.3 奖励函数（Reward）

`MpcWeightEnvReward` 将奖励拆成多部分，便于调参与诊断：

| 分量               | 含义                     | 默认权重   | 说明 |
|--------------------|--------------------------|------------|------|
| height             | 与目标高度偏差的负惩罚   | w_height=1.0 | 目标高度默认 0.75 m |
| velocity           | 与目标 vx、vy 偏差的负惩罚 | w_velocity=0.5 | 默认目标速度为 0 |
| orientation        | roll/pitch 偏离的负惩罚  | w_orientation=0.5 | 保持躯干直立 |
| action_magnitude   | \|\|a\|\|² 的负惩罚      | 0.01       | 限制权重大幅偏离 |
| action_smooth      | \|\|a - a_prev\|\|² 的负惩罚 | 0.02    | 鼓励权重平滑 |
| fall               | 高度低于阈值时的大负奖励 | -10.0      | 与 terminated 同时触发 |

- **终止条件**：基座高度 < `fall_height_threshold`（默认 0.35 m）时 `terminated=True`，episode 结束。
- **自定义奖励**：可传入自定义 `MpcWeightEnvReward` 子类或新实例，在构造 `G1MpcWeightEnv` 时通过 `reward_fn=...` 传入；也可修改默认参数（如 `target_height`、`target_vx`、各 `w_*`）。

---

## 4. 环境 API 与基本用法

### 4.1 创建环境

```python
import os
from test_mpc_RL import make_g1_mpc_weight_env, MpcWeightEnvReward

# 最小示例（默认多速率 RL 50 / MPC 100 / Sim 1000 Hz）
env = make_g1_mpc_weight_env(
    headless=True,
    seed=42,
)

# 自定义多速率（可选）
env = make_g1_mpc_weight_env(
    rl_frequency_hz=50.0,
    mpc_frequency_hz=100.0,
    sim_frequency_hz=1000.0,
    headless=True,
    seed=42,
)

# 自定义速度指令与奖励（例如希望向前 0.2 m/s）
velocity_command = np.array([0.2, 0.0, 0.75, 0.0])  # vx, vy, height, vyaw
reward_fn = MpcWeightEnvReward(
    target_height=0.75,
    target_vx=0.2,
    target_vy=0.0,
    w_velocity=0.8,
    fall_height_threshold=0.35,
)
env = make_g1_mpc_weight_env(
    headless=True,
    velocity_command=velocity_command,
    reward_fn=reward_fn,
    seed=42,
)
```

需要更细粒度配置（如自定义配置文件路径）时，可直接使用 `G1MpcWeightEnv` 的完整构造函数，传入与上面相同的参数即可。

**多速率参数**（可选）：
- `rl_frequency_hz`：RL 步频，默认 50 Hz（每步 0.02 s）。
- `mpc_frequency_hz`：MPC 求解与关节控制更新频率，默认 100 Hz；需满足 `rl_dt >= mpc_dt`（即 `rl_frequency_hz <= mpc_frequency_hz`）。
- `sim_frequency_hz`：MuJoCo 仿真步频，默认 1000 Hz；需满足 `mpc_dt >= sim_dt`（即 `sim_frequency_hz >= mpc_frequency_hz`）。
- 每 RL 步内 MPC 次数 = `rl_dt / mpc_dt`，每次 MPC 后仿真步数 = `mpc_dt / sim_dt`。

### 4.2 reset / step

- **reset(seed=None, options=None)**
  - 返回：`(obs, info)`。
  - 将仿真与 MPC 重置到初始站立姿态，并返回初始观测与 `info`（如 `sim_time`、`step`）。

- **step(action)**
  - 输入：`action` 为长度 58 的数组（多余会被截断、不足会补 0，建议显式裁剪到 `env.action_space.clip(action)`）。
  - 返回：`(obs, reward, terminated, truncated, info)`。
  - `info` 中包含 `height`、`fallen`、`reward_components` 等，便于记录与调试。

示例循环：

```python
obs, info = env.reset(seed=42)
for _ in range(500):
    action = env.action_space.sample()  # 或由策略输出
    action = env.action_space.clip(action)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

---

## 5. 与常见 RL 库对接

### 5.1 Gymnasium 兼容性

环境遵循 Gymnasium 的 `reset` / `step` 约定（含 `terminated`、`truncated`），但**未继承 `gymnasium.Env`**，因此：

- **observation_space / action_space**：
  - `env.observation_space` 为带有 `shape`、`dtype` 的简单对象；
  - `env.action_space` 为 `MpcResidualActionSpace`（含 `dim`、`low`、`high`、`sample()`、`clip()`）。

若算法要求标准的 `gymnasium.spaces.Box`，可做一层薄封装：

```python
import gymnasium as gym
import numpy as np

class G1MpcWeightGymEnv(gym.Env):
    def __init__(self, **kwargs):
        super().__init__()
        from test_mpc_RL import G1MpcWeightEnv
        self._env = G1MpcWeightEnv(**kwargs)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=self._env.observation_space.shape,
            dtype=self._env.observation_space.dtype,
        )
        self.action_space = gym.spaces.Box(
            low=self._env.action_space.low,
            high=self._env.action_space.high,
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        return self._env.reset(seed=seed, options=options)

    def step(self, action):
        return self._env.step(self._env.action_space.clip(action))

    @property
    def unwrapped(self):
        return self._env.unwrapped
```

### 5.2 Stable-Baselines3 (SB3)

- 使用上述 `G1MpcWeightGymEnv` 包装后，即可传入 `PPO`、`SAC` 等。
- 动作空间为 `Box(58,)`，策略输出建议裁剪到 `[-1, 1]`（SB3 的 `clip` 需与 `action_space` 一致，这里已用 `Box(low, high)` 定义）。
- 示例（PPO）：

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

def make_env():
    return G1MpcWeightGymEnv(headless=True, seed=None)

env = DummyVecEnv([make_env])
model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=dict(net_arch=[256, 256]),
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    verbose=1,
)
model.learn(total_timesteps=100_000, callback=[
    CheckpointCallback(save_freq=10000, name_prefix="g1_mpc_weight"),
])
model.save("g1_mpc_weight_ppo")
```

其中 `G1MpcWeightGymEnv` 的 `__init__` 使用 `self._env = G1MpcWeightEnv(**kwargs)` 即可，与上文 5.1 节封装一致。

### 5.3 多进程/向量化

- 每个环境实例会创建自己的 MPC 接口与 MuJoCo 仿真，**进程/线程开销较大**，建议先用 `DummyVecEnv` 小规模验证，再视需要改用 `SubprocVecEnv` 并控制并行数量（如 4～8），避免内存与 CPU 过载。
- 无头运行（`headless=True`）可减少图形相关报错与资源占用。

---

## 6. 训练建议与超参数

### 6.1 奖励与任务

- **站立/慢走**：可设 `target_vx=0`，适当加大 `w_height`、`w_orientation`，便于先学稳。
- **前进**：逐步提高 `target_vx`（如 0.1 → 0.2），并提高 `w_velocity`，观察 `reward_components` 中各项是否合理。
- **跌倒惩罚**：保持 `fall_height_threshold` 与 `reward_fn.fall` 一致，避免过早终止时可适当提高阈值（如 0.4 m）做早期安全训练。

### 6.2 动作与探索

- 动作范围已为 `[-1, 1]`，一般无需再乘系数；若发现权重变化过猛，可适当增大 `w_action_magnitude` 和 `w_action_smooth`。
- 若用 PPO，可适当降低初始 `ent_coef`，避免探索过大导致 MPC 权重剧烈抖动。

### 6.3 步长与 episode 长度

- 每 RL 步对应 0.02 s 仿真时间（50 Hz），单 episode 若 500～1000 步即 10～20 s 仿真时长，可按需要设 `max_episode_steps`（在封装层做 `truncated`）以控制单次 rollout 长度与 batch 大小。

### 6.4 观测归一化

- 若训练方差大，可对观测做标准化（如 RunningMeanStd）或按固定范围裁剪后再输入策略，有助于稳定。

---

## 7. 常见问题

**Q: 导入 `humanoid_wb_mpc_py` 失败**
A: 确认已在 `wb_humanoid_mpc_ws` 下编译并安装，且 Python 能找到安装目录下的 `.so`（脚本中默认加入 `/wb_humanoid_mpc_ws/install/humanoid_wb_mpc/lib`，可按实际安装路径修改）。

**Q: 无头模式下报 OpenGL/EGL 错**
A: 确保已安装 EGL 相关库，并设置 `HEADLESS=1`、`MUJOCO_GL=egl`（脚本会在无 DISPLAY 时自动设置）。

**Q: 环境步进很慢**
A: 每 RL 步包含 2 次 MPC 求解与 20 次仿真步（默认 50/100/1000 Hz），单步耗时远大于纯 MuJoCo；可先减小 `rl_frequency_hz`（如 25 Hz）或减少并行环境数以加快训练。

**Q: 如何只训练「部分权重」？**
A: 当前接口为 58 维整体传入；若希望部分维度固定，可在策略输出后手动将对应维度置 0（或固定值），再调用 `set_residual_weights`。

**Q: 如何记录/可视化 reward 分量？**
A: 每步 `info["reward_components"]` 为字典（如 `height`、`velocity`、`orientation` 等），在 callback 或 logger 中记录即可。

---

## 8. 文件与脚本索引

- **环境实现**：`humanoid_nmpc/humanoid_wb_mpc/scripts/test_mpc_RL.py`
  - `G1MpcWeightEnv`、`make_g1_mpc_weight_env`、`MpcResidualActionSpace`、`MpcWeightEnvReward`。
- **快速测试**：在 `test_mpc_RL.py` 目录下执行
  `python3 test_mpc_RL.py`
  会跑一段随机策略并打印步数、高度与奖励。
- **C++ 权重映射**：`MpcWeightAdjustmentModule`（`set_residual_weights` → `Q_i = Q_base_i * exp(a_i)`）在
  `humanoid_wb_mpc/src/synchronized_module/MpcWeightAdjustmentModule.cpp`。

按上述说明即可基于本环境进行「RL 调节 MPC 权重」的强化学习训练；具体算法与超参数可根据任务再细调。
