"""
G1 MPC 权重 PPO 训练脚本

使用包内配置与环境：默认使用 G1MpcWeightEnv（仿真同步多速率），
也可切换为 G1MpcEnv（简化 MuJoCo 步进）。路径从 humanoid_wb_mpc.config 读取。

用法速记（常用环境变量）：

- 有头/无头模式：
  - 有头（打开 MuJoCo 视图）：HEADLESS=0
  - 无头（推荐训练吞吐）：HEADLESS=1

- 训练步数与 TensorBoard：
  - TOTAL_TIMESTEPS=200000      # 总训练步数（SB3 的 timesteps）
  - TB_LOG_DIR=./ppo_g1_logs/... # TensorBoard 日志目录

- 将 reset 随机化（domain randomization）采样写入 TensorBoard：
  - LOG_RESET_RAND=1            # 开启 domain_rand/* 标量写入
  - LOG_RESET_RAND_EVERY=20     # 每隔多少个 episode 在终端打印一次（不影响写 TB）
  - 说明：domain_rand/* 来自环境 reset 时采样的 mu/外力/velocity_command 等；
         仅用于确认随机化是否生效，不改变训练逻辑。

- 修改训练超参（不改代码，直接用环境变量覆盖）：
  - LEARNING_RATE=1e-4
  - N_STEPS=2048
  - BATCH_SIZE=128
  - N_EPOCHS=10
  - GAMMA=0.99
  - CLIP_RANGE=0.1
  - ENT_COEF=0.01
  - TARGET_KL=0.01
  - SEED=42

示例：
  # 有头训练 + 写入 domain_rand/* 到 TensorBoard
  LOG_RESET_RAND=1 HEADLESS=0 TOTAL_TIMESTEPS=100000 TB_LOG_DIR=./ppo_g1_logs/phase1_short python3 train_g1_ppo_v1.py

  # 无头训练 + 调学习率/rollout 长度
  HEADLESS=1 LEARNING_RATE=5e-5 N_STEPS=4096 TOTAL_TIMESTEPS=200000 python3 train_g1_ppo_v1.py
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 在导入 humanoid_wb_mpc 前加入 C++ 扩展路径（否则包 __init__ -> core 会先 import humanoid_wb_mpc_py 报错）
_ws = os.environ.get("WB_HUMANOID_MPC_WS", "/wb_humanoid_mpc_ws")
_install = os.path.join(_ws, "install", "humanoid_wb_mpc")
_py_ver = "python{}.{}".format(sys.version_info.major, sys.version_info.minor)
for _candidate in (
    os.path.join(_install, "lib", _py_ver, "site-packages"),
    os.path.join(_install, "lib"),
):
    if os.path.isdir(_candidate) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)
        break
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl" if os.environ.get("HEADLESS", "0") == "1" or not os.environ.get("DISPLAY") else "glfw"

import humanoid_wb_mpc.bootstrap  # noqa: F401
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import TimeLimit
from humanoid_wb_mpc.config import (
    DEFAULT_TASK_FILE,
    DEFAULT_URDF_FILE,
    DEFAULT_REF_FILE,
)
from humanoid_wb_mpc import make_g1_mpc_weight_env

# 是否使用完整仿真环境（G1MpcWeightEnv）；False 则使用简化 G1MpcEnv（会导入 mujoco，需 GL）
USE_WEIGHT_ENV = True


class ResetRandDebugCallback(BaseCallback):
    """
    在 episode 第 1 个 step 打印/记录 reset 随机化采样值，
    用于确认外力/摩擦/初始姿态/velocity_command 是否确实在生效。

    启用:
      LOG_RESET_RAND=1
    可选:
      LOG_RESET_RAND_EVERY=20   # 每隔多少个 episode 打印一次（默认 20）
    """

    def __init__(self, every: int = 20, verbose: int = 0):
        super().__init__(verbose=verbose)
        self._every = max(1, int(every))
        self._reset_seen = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if not infos:
            return True

        for info in infos:
            if not isinstance(info, dict):
                continue
            if info.get("step", None) != 1:
                continue
            if "reset_rand/enabled" not in info:
                continue

            self._reset_seen += 1

            mu = info.get("reset_rand/mu", None)
            force = info.get("reset_rand/force", None) or {}
            vel_cmd = info.get("reset_rand/vel_cmd", None)

            # 写入 TensorBoard（如果启用了 tensorboard_log）
            if mu is not None:
                self.logger.record("domain_rand/mu", float(mu))
            self.logger.record("domain_rand/force_applied", 1.0 if force else 0.0)
            if force:
                self.logger.record("domain_rand/force_fx", float(force.get("fx", 0.0)))
                self.logger.record("domain_rand/force_fy", float(force.get("fy", 0.0)))
                self.logger.record("domain_rand/force_fz", float(force.get("fz", 0.0)))
                self.logger.record("domain_rand/force_duration_steps", float(force.get("duration_steps", 0.0)))
            if vel_cmd is not None and len(vel_cmd) >= 4:
                self.logger.record("domain_rand/cmd_vx", float(vel_cmd[0]))
                self.logger.record("domain_rand/cmd_vy", float(vel_cmd[1]))
                self.logger.record("domain_rand/cmd_height", float(vel_cmd[2]))
                self.logger.record("domain_rand/cmd_yaw_rate", float(vel_cmd[3]))

            # 控制台打印（降频）
            if self._reset_seen % self._every == 0:
                print(f"[domain_rand] ep={self._reset_seen} mu={mu} force={'yes' if force else 'no'} vel_cmd={vel_cmd}")

        return True


class RobustnessMetricsCallback(BaseCallback):
    """
    记录/观察鲁棒性指标（写入 TensorBoard）：

    - robust/terminated: 本 episode 是否跌倒终止（1/0）
    - robust/fall_rate_ema: 跌倒率（指数滑动平均）
    - robust/recovery_time_s: 受推扰后恢复到目标高度/姿态的时间（秒）
    - robust/base_drift_xy_m: episode 结束时基座水平漂移（米，粗略作为滑移代理指标）

    启用:
      LOG_ROBUST_METRICS=1
    可选阈值:
      REC_H_EPS=0.03           # 高度误差阈值（m）
      REC_ANGLE_EPS=0.12       # roll/pitch 误差阈值（rad）
      REC_HOLD_STEPS=10        # 连续满足阈值的 RL 步数才算恢复
      FALL_RATE_EMA_ALPHA=0.05 # 跌倒率 EMA 的 alpha
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose=verbose)
        self._ep_idx = 0
        self._fall_rate_ema = 0.0

        self._cur_has_push = False
        self._cur_t0_push = None
        self._cur_recovered = False
        self._cur_hold = 0
        self._cur_init_xy = None

        self._h_eps = float(os.environ.get("REC_H_EPS", "0.03"))
        self._a_eps = float(os.environ.get("REC_ANGLE_EPS", "0.12"))
        self._hold_steps = int(os.environ.get("REC_HOLD_STEPS", "10"))
        self._ema_alpha = float(os.environ.get("FALL_RATE_EMA_ALPHA", "0.05"))

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None) or []
        dones = self.locals.get("dones", None) or []
        # SB3: dones 是 np.ndarray/bool list，infos 同长度

        for i, info in enumerate(infos):
            if not isinstance(info, dict):
                continue

            step = info.get("step", None)
            if step == 1:
                force = info.get("reset_rand/force", None) or {}
                self._cur_has_push = bool(force)
                self._cur_t0_push = float(info.get("sim_time", 0.0)) if self._cur_has_push else None
                self._cur_recovered = False
                self._cur_hold = 0
                base_pos = info.get("reset_rand/base_pos", None)
                if base_pos is not None and len(base_pos) >= 2:
                    self._cur_init_xy = (float(base_pos[0]), float(base_pos[1]))
                else:
                    self._cur_init_xy = None

            # 恢复时间：只在“有推扰”的 episode 里计算
            if self._cur_has_push and (not self._cur_recovered):
                vel_cmd = info.get("reset_rand/vel_cmd", None)
                target_h = float(vel_cmd[2]) if vel_cmd is not None and len(vel_cmd) >= 3 else 0.75

                h = float(info.get("height", 0.0))
                euler = info.get("root_euler_zyx", None)  # [yaw,pitch,roll] in zyx order
                if euler is not None and len(euler) >= 3:
                    pitch = float(euler[1])
                    roll = float(euler[2])
                else:
                    pitch, roll = 0.0, 0.0

                ok = (abs(h - target_h) <= self._h_eps) and (abs(roll) <= self._a_eps) and (abs(pitch) <= self._a_eps)
                if ok:
                    self._cur_hold += 1
                    if self._cur_hold >= self._hold_steps:
                        t_now = float(info.get("sim_time", 0.0))
                        t0 = self._cur_t0_push if self._cur_t0_push is not None else t_now
                        self._cur_recovered = True
                        self.logger.record("robust/recovery_time_s", max(0.0, t_now - t0))
                else:
                    self._cur_hold = 0

            done = bool(dones[i]) if i < len(dones) else False
            if done:
                self._ep_idx += 1
                terminated = 1.0 if bool(info.get("fallen", False)) else 0.0
                self.logger.record("robust/terminated", terminated)
                self._fall_rate_ema = (1.0 - self._ema_alpha) * self._fall_rate_ema + self._ema_alpha * terminated
                self.logger.record("robust/fall_rate_ema", float(self._fall_rate_ema))

                # 水平漂移（粗略滑移代理）：episode 结束时 root_pos - init_xy
                root_pos = info.get("root_pos", None)
                if self._cur_init_xy is not None and root_pos is not None and len(root_pos) >= 2:
                    dx = float(root_pos[0]) - float(self._cur_init_xy[0])
                    dy = float(root_pos[1]) - float(self._cur_init_xy[1])
                    self.logger.record("robust/base_drift_xy_m", float((dx * dx + dy * dy) ** 0.5))

        return True


def main():
    headless = os.environ.get("HEADLESS", "1") == "1"
    seed = int(os.environ.get("SEED", "42"))

    if USE_WEIGHT_ENV:
        print(f"正在初始化 G1 MPC 权重环境（仿真同步多速率）...(headless={headless})")
        env = make_g1_mpc_weight_env(
            task_file=DEFAULT_TASK_FILE,
            urdf_file=DEFAULT_URDF_FILE,
            ref_file=DEFAULT_REF_FILE,
            headless=headless,
            seed=seed,
        )
    else:
        from humanoid_wb_mpc.envs import G1MpcEnv
        print("正在初始化 G1 MPC 强化学习环境（简化）...")
        env = G1MpcEnv(DEFAULT_TASK_FILE, DEFAULT_URDF_FILE, DEFAULT_REF_FILE)

    max_episode_steps = 2048
    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    total_steps = int(os.environ.get("TOTAL_TIMESTEPS", 200_000))
    tb_log_dir = os.environ.get("TB_LOG_DIR", "./ppo_g1_logs/phase1_short")

    # PPO 超参（可用环境变量覆盖）
    learning_rate = float(os.environ.get("LEARNING_RATE", "1e-4"))
    n_steps = int(os.environ.get("N_STEPS", "2048"))
    batch_size = int(os.environ.get("BATCH_SIZE", "128"))
    n_epochs = int(os.environ.get("N_EPOCHS", "10"))
    gamma = float(os.environ.get("GAMMA", "0.99"))
    clip_range = float(os.environ.get("CLIP_RANGE", "0.1"))
    ent_coef = float(os.environ.get("ENT_COEF", "0.01"))
    target_kl = float(os.environ.get("TARGET_KL", "0.01"))

    policy_kwargs = dict(net_arch=[dict(pi=[256, 256], qf=[256, 256])])
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=tb_log_dir,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        clip_range=clip_range,
        ent_coef=ent_coef,
        target_kl=target_kl,
        policy_kwargs=policy_kwargs,
        device="auto",
        seed=seed,
    )

    # 第一阶段短训练试跑：1e5～2e5 steps（单 env），用于观察 reward / episode length 曲线
    print("🚀 启动训练: G1 步态残差优化（Phase1 短训练）...")
    print(f"   总步数: {total_steps:,} | TensorBoard: {tb_log_dir}")
    callback = None
    if os.environ.get("LOG_RESET_RAND", "0") == "1":
        every = int(os.environ.get("LOG_RESET_RAND_EVERY", "20"))
        callback = ResetRandDebugCallback(every=every)
        print(f"   reset 随机化日志已开启: LOG_RESET_RAND_EVERY={every}")
    if os.environ.get("LOG_ROBUST_METRICS", "0") == "1":
        # 允许和 LOG_RESET_RAND 同时开；SB3 支持 callback list
        robust_cb = RobustnessMetricsCallback()
        callback = [cb for cb in (callback, robust_cb) if cb is not None]
        print("   鲁棒性指标记录已开启: LOG_ROBUST_METRICS=1")
    model.learn(total_timesteps=total_steps, progress_bar=True, callback=callback)
    save_name = f"g1_ppo_residual_v1_phase1_{total_steps}"
    model.save(save_name)
    print(f"✅ 训练模型已保存为 {save_name}.zip")
    tb_abs = os.path.abspath(tb_log_dir)
    print(f"   查看曲线: tensorboard --logdir {tb_abs}")
    print(f"   （或先 cd 到本脚本所在目录再: tensorboard --logdir {tb_log_dir}）")


if __name__ == "__main__":
    main()
