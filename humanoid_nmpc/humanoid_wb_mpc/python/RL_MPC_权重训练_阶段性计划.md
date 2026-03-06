# G1 RL 调节 MPC 权重 — 阶段性计划书

本文档描述「RL 调节 MPC 权重」训练的两阶段路线图：先单 env / 少量并行、保证网络学得明白且底层算得安全，再在第二阶段推进大量并行与训练吞吐。

---

## 一、总体思路

从控制目标上，本项目希望通过「RL 调节 MPC 权重」实现两个渐进式能力：**（1）站立时的抗干扰能力**（外力推扰后能迅速稳住、不跌倒）、**（2）行走时对环境的适应能力**（不同地面、不同速度指令仍能平稳行走）。在阶段划分上，**第一阶段优先聚焦站立与外力扰动场景**，确保基础平衡鲁棒；**第二阶段在此基础上再引入行走与环境随机化**。

| 阶段 | 目标 | 并行规模 | 架构侧重 |
|------|------|----------|----------|
| **第一阶段** | 网络能学得起来；MPC/仿真稳定、安全 | 单 env → 4～8 env | 保持当前 Pybind11 + Python 拼装，优先调奖励与策略 |
| **第二阶段** | 提升样本效率与训练吞吐 | 数十～上百 env | 可选：C++ 向量化 env，减少 Python↔C++ 边界与 GIL 影响 |

当前代码已支持第一阶段：环境与配置集中在 `humanoid_wb_mpc` 包内（`envs/G1MpcWeightEnv`、`config`、`utils/reward_handler` 等），训练入口为 `scripts/train_g1_ppo_v1.py`，测试入口为 `scripts/test_mpc_rl.py`。

---

## 二、第一阶段：单 env / 少量并行

**核心目标**：让策略在单 env 或少量并行下**学得明白**，并让底层 MPC 与仿真**算得安全**。

### 2.1 阶段目标与验收标准

- **学得明白（站立场景）**：在以站立平衡为主的任务设置下，默认或轻度调参即可看到明显学习信号（episode 回报上升、episode 长度增加、跌倒率下降）。
- **算得安全**：MPC 不频繁不可行、无异常关节/基座指令；仿真与控制器在长时间运行下稳定，无偶发崩溃。
- **可复现**：固定种子下训练/评估结果可复现；配置与路径集中管理，便于后续扩展。

### 2.2 任务清单与优先级

#### 2.2.1 优先：打通「能学、能训」闭环

| 序号 | 任务 | 说明 | 产出/验收 |
|------|------|------|-----------|
| 0 | 站立平衡任务定义 | 在 env/config 中关闭或弱化行走相关奖励与指令，设置零/低速速度命令，仅考察站立保持与恢复 | 明确站立 reward/终止条件，为后续行走扩展预留接口 |
| 1 | 短训练试跑 | 使用 `train_g1_ppo_v1.py` 跑约 1e5～2e5 steps（单 env） | 得到 reward / episode length 曲线（如 TensorBoard） |
| 2 | 曲线诊断 | 观察 reward 是否持续极负、是否几乎每步 terminated、height 是否长期低于目标 | 明确是「学不动」还是「奖励/终止设计」问题 |
| 3 | 奖励与终止调参 | 在 `MpcWeightEnvReward` 中调整 `w_height`、`w_velocity`、`fall_height_threshold` 等；必要时加存活奖励或放宽终止条件 | 单 env 下出现明显学习趋势 |
| 4 | 策略超参 | 根据 58 维连续动作与多速率特性，微调 PPO 的 `n_steps`、`batch_size`、`learning_rate`、`ent_coef` 等 | 训练稳定、不爆炸不塌缩 |

#### 2.2.2 其次：底层安全与鲁棒性

| 序号 | 任务 | 说明 | 产出/验收 |
|------|------|------|-----------|
| 5 | MPC 可行性监控 | 观察训练/评估中 MPC 是否经常无解或超时；必要时在 C++ 或任务配置中加强约束/限幅或 fallback | 无大量不可行或异常输出 |
| 6 | 单步耗时与超时 | 利用现有 SQP 日志关注 LQ/QP 时间；若偶发单步极慢，可加超时或降频保护 | 避免单步卡死影响训练 |
| 7 | 轻量随机化（可选） | 在 `G1MpcWeightEnv.reset` 中对初始高度/姿态、`velocity_command` 以及外力扰动（如短脉冲推力）、地面参数（摩擦系数等）做小范围随机，提升站立鲁棒性 | 策略从小适应不确定性与外界扰动，为 Phase 2 行走与复杂地形打基础 |

#### 2.2.3 过渡：少量并行验证

| 序号 | 任务 | 说明 | 产出/验收 |
|------|------|------|-----------|
| 8 | 4～8 个并行 env | 使用 SubprocVecEnv 或等价多进程，每进程一个 `G1MpcWeightEnv`，不改 C++ | 多实例下 MPC/仿真稳定，资源可接受 |
| 9 | 样本效率对比 | 对比单 env 与 4～8 env 在相同 wall-clock 下的提升幅度 | 确认少量并行带来收益，为 Phase 2 规模提供参考 |

### 2.3 为第二阶段预留的设计

- **环境接口**：保持 Gymnasium 风格 `reset()` / `step(action)`；若 Phase 2 做 C++ 向量化 env，可设计 `reset_batch()` / `step_batch(actions)` 返回批量 obs/reward/done，Python 侧做薄 VecEnv 封装。
- **配置与种子**：路径、频率、种子等集中在 `humanoid_wb_mpc.config` 与 env 构造参数中，便于将来 C++ 向量化 env 复用同一套配置。
- **不在本阶段实现 C++ 向量化**：Phase 1 仅使用现有「单 env + 少量多进程 env」，把「学得明白、算得安全」做到位后再启动 Phase 2。

### 2.4 第一阶段完成标志

- 单 env 训练曲线显示稳定学习（reward 上升、episode 变长）。
- 4～8 个并行 env 可稳定跑完较长训练，无频繁崩溃或数值异常。
- 奖励、终止与超参有文档或注释记录，便于复现与迭代。

---

## 三、第二阶段：大量并行与训练吞吐

**核心目标**：在 Phase 1 的站立抗干扰策略与底层均稳定的前提下，引入行走任务与环境随机化，并通过**大量并行 env** 提升样本效率与训练吞吐。

### 3.1 阶段目标与验收标准

- 支持数十～上百个 env 并行，且单机或分布式下训练吞吐明显优于「多进程 × 单 env」。
- 在包含行走与环境随机化（坡度、摩擦、轻微不平整等）的任务下，策略能保持稳定行走与恢复能力。
- 与 Phase 1 的策略/奖励/接口兼容，可加载 Phase 1 检查点继续训练或评估。

### 3.2 任务清单（概要）

| 序号 | 任务 | 说明 |
|------|------|------|
| 1 | 性能瓶颈评估 | 用 8～16 env 多进程测 CPU/内存与 GIL 影响，确认 Python 调度是否成为瓶颈 |
| 2 | C++ 向量化 env 设计 | 设计 `VecG1MpcWeightEnv`（或等价）：N 份仿真 + N 份 MPC（或共享部分结构），提供 `step_batch(actions)` → `obs_batch, rewards, dones, infos` |
| 3 | Pybind11 批量接口 | 暴露批量 step/reset，仅做少量 Python↔C++ 边界与批量数组交换，供 SB3 等 VecEnv 封装 |
| 4 | Python VecEnv 封装 | 用新 C++ 批量接口实现 VecEnv，替换 SubprocVecEnv，对接现有训练脚本与 config |
| 5 | 吞吐与稳定性验证 | 对比 Phase 1 多进程与 Phase 2 向量化 env 的 steps/s 与收敛速度 |

### 3.3 第二阶段启动前提

- 第一阶段已完成验收，单 env 与少量并行均「学得明白、算得安全」。
- 有明确需求（如更大 batch、更短训练时间）或观测到多进程扩展性不足时再投入 C++ 向量化开发。

---

## 四、与现有文档、代码的对应关系

| 内容 | 位置 |
|------|------|
| 环境 API、奖励/观测/动作说明、编译与运行 | `humanoid_nmpc/humanoid_wb_mpc/scripts/README_RL_MPC_WEIGHT_TRAINING.md` |
| 环境与配置实现 | `humanoid_nmpc/humanoid_wb_mpc/python/humanoid_wb_mpc/`（`envs/G1MpcWeightEnv.py`、`config/`、`utils/reward_handler.py`、`spaces.py`） |
| 训练入口 | `humanoid_nmpc/humanoid_wb_mpc/python/scripts/train_g1_ppo_v1.py` |
| 环境测试入口 | `humanoid_nmpc/humanoid_wb_mpc/python/scripts/test_mpc_rl.py` |
| 阶段性计划（本文档） | `humanoid_nmpc/humanoid_wb_mpc/python/RL_MPC_权重训练_阶段性计划.md` |

---

## 五、建议的下一步（立即执行）

1. 在 env/config 中先固定为**站立平衡场景**（无行走速度指令或仅极低速度），然后**跑一次短训练**（约 1e5 steps），保存 TensorBoard 或曲线数据，观察 reward 与 episode length。
2. **针对站立保持与外力扰动恢复**，调整 `MpcWeightEnvReward` 与 `fall_height_threshold` 等，直到单 env 站立任务能明显「学得动」且对小扰动有恢复能力。
3. **（可选）** 在 `G1MpcWeightEnv.reset` 中加入轻度状态/外力/地面参数随机化，观察站立鲁棒性与训练稳定性。
4. 当站立抗干扰效果稳定后，再逐步打开行走速度指令与简单环境随机化（如轻微坡度），并在此基础上启用 4～8 个并行 env；若多并行仍稳定，再规划第二阶段「大规模并行 + 行走环境适应」的具体排期与实现细节。

---

*文档版本：1.0 | 与当前 python 包及 train_g1_ppo_v1 / test_mpc_rl 脚本对应*
