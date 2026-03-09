"""
G1 MPC 权重 PPO — 站立抗扰动导向（加大扰动）

专用于「纯站立 + 加强外力扰动」：从 Phase 1 的 50 万步模型续训，提高外力概率与力度，
不引入行走。与 train_g1_ppo_v1_行走导向.py（Phase 2）可两台机器并行跑。

- 任务：纯站立（vel_cmd=0），只调 base 相关 Q 维（PHASE1_ACTIVE_DIMS）。
- 加大扰动：FORCE_PROB 默认 0.85（Phase 1 为 0.70），FORCE_MAG 默认 45~100 N（Phase 1 为 30~80）。
- 初始状态随机化（base/关节/摩擦）仍由 G1MpcWeightEnv 内默认范围控制，与 Phase 1 相同。

常用环境变量：
  - LOAD_MODEL=ppo_zip/phase1/g1_ppo_residual_v1_phase1_500000_n4.zip  # 默认即此，从 50 万步续训
  - TB_LOG_DIR=./ppo_g1_logs/phase1_heavy_disturb   # 默认日志目录
  - TOTAL_TIMESTEPS=200000
  - N_ENVS=4 或 8，HEADLESS=1
  - FORCE_PROB=0.85   FORCE_MAG_MIN=45   FORCE_MAG_MAX=100   # 可覆盖
  - LOG_ROBUST_METRICS=1   # 记录 fall_rate_ema、base_drift 等

示例（另一台机器）：
  cd scripts
  LOAD_MODEL=ppo_zip/phase1/g1_ppo_residual_v1_phase1_500000_n4.zip \\
  N_ENVS=4 HEADLESS=1 TOTAL_TIMESTEPS=200000 \\
  TB_LOG_DIR=./ppo_g1_logs/phase1_heavy_disturb \\
  python3 train_g1_ppo_v1_站立抗扰动导向.py
"""

import os
import sys

# 段错误时打印 Python 调用栈（便于定位 C 扩展崩溃点）
try:
    import faulthandler
    faulthandler.enable(all_threads=True)
except Exception:
    pass

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

import numpy as np
import humanoid_wb_mpc.bootstrap  # noqa: F401
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from gymnasium.wrappers import TimeLimit
from humanoid_wb_mpc.config import (
    DEFAULT_TASK_FILE,
    DEFAULT_URDF_FILE,
    DEFAULT_REF_FILE,
)
from humanoid_wb_mpc import make_g1_mpc_weight_env
from humanoid_wb_mpc.utils.reward_handler import MpcWeightEnvReward

# ─── 站立任务 active Q 维（与 Phase 1 一致）──────────────────────────────────
# 只调 base 相关维：高度、roll/pitch、base 线/角速度
STANDING_ACTIVE_DIMS = [2, 4, 5, 29, 30, 31, 32, 33, 34]

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
        _infos_r = self.locals.get("infos", None)
        infos = list(_infos_r) if _infos_r is not None else []
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
        # 以下状态改为按 env 索引存储，避免多 env（SubprocVecEnv）时相互覆盖
        self._fall_rate_ema: dict = {}
        self._cur_has_push: dict = {}
        self._cur_t0_push: dict = {}
        self._cur_recovered: dict = {}
        self._cur_hold: dict = {}
        self._cur_init_xy: dict = {}

        self._h_eps = float(os.environ.get("REC_H_EPS", "0.03"))
        self._a_eps = float(os.environ.get("REC_ANGLE_EPS", "0.12"))
        self._hold_steps = int(os.environ.get("REC_HOLD_STEPS", "10"))
        self._ema_alpha = float(os.environ.get("FALL_RATE_EMA_ALPHA", "0.05"))

    def _on_step(self) -> bool:
        _infos = self.locals.get("infos", None)
        infos = list(_infos) if _infos is not None else []
        _dones = self.locals.get("dones", None)
        dones = list(_dones) if _dones is not None else []

        for i, info in enumerate(infos):
            if not isinstance(info, dict):
                continue

            # 懒初始化：首次遇到该 env 索引时建立独立状态，避免多 env 相互覆盖
            if i not in self._cur_has_push:
                self._fall_rate_ema[i] = 0.0
                self._cur_has_push[i] = False
                self._cur_t0_push[i] = None
                self._cur_recovered[i] = False
                self._cur_hold[i] = 0
                self._cur_init_xy[i] = None

            step = info.get("step", None)
            if step == 1:
                force = info.get("reset_rand/force", None) or {}
                self._cur_has_push[i] = bool(force)
                self._cur_t0_push[i] = float(info.get("sim_time", 0.0)) if self._cur_has_push[i] else None
                self._cur_recovered[i] = False
                self._cur_hold[i] = 0
                base_pos = info.get("reset_rand/base_pos", None)
                if base_pos is not None and len(base_pos) >= 2:
                    self._cur_init_xy[i] = (float(base_pos[0]), float(base_pos[1]))
                else:
                    self._cur_init_xy[i] = None

            # 恢复时间：只在"有推扰"的 episode 里计算
            if self._cur_has_push[i] and (not self._cur_recovered[i]):
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
                    self._cur_hold[i] += 1
                    if self._cur_hold[i] >= self._hold_steps:
                        t_now = float(info.get("sim_time", 0.0))
                        t0 = self._cur_t0_push[i] if self._cur_t0_push[i] is not None else t_now
                        self._cur_recovered[i] = True
                        self.logger.record("robust/recovery_time_s", max(0.0, t_now - t0))
                else:
                    self._cur_hold[i] = 0

            done = bool(dones[i]) if i < len(dones) else False
            if done:
                self._ep_idx += 1
                terminated = 1.0 if bool(info.get("fallen", False)) else 0.0
                self.logger.record("robust/terminated", terminated)
                self._fall_rate_ema[i] = (1.0 - self._ema_alpha) * self._fall_rate_ema[i] + self._ema_alpha * terminated
                # 上报所有已初始化 env 的 EMA 均值，消除单次 done 事件对全局指标的扭曲
                mean_ema = sum(self._fall_rate_ema.values()) / len(self._fall_rate_ema)
                self.logger.record("robust/fall_rate_ema", mean_ema)

                # 水平漂移（粗略滑移代理）：episode 结束时 root_pos - init_xy
                root_pos = info.get("root_pos", None)
                if self._cur_init_xy[i] is not None and root_pos is not None and len(root_pos) >= 2:
                    dx = float(root_pos[0]) - float(self._cur_init_xy[i][0])
                    dy = float(root_pos[1]) - float(self._cur_init_xy[i][1])
                    self.logger.record("robust/base_drift_xy_m", float((dx * dx + dy * dy) ** 0.5))

        return True


def _make_single_env(
    rank: int,
    seed: int,
    headless: bool,
    max_episode_steps: int,
    velocity_command=None,
    reward_fn=None,
    active_weight_dims=None,
    force_prob: float = 0.45,
    force_mag_range=(28.0, 62.0),
    vel_cmd_rand_range=(0.05, 0.05, 0.04),
):
    """子进程/单进程内构造单个 env，用于 VecEnv 或单 env。"""
    def _init():
        e = make_g1_mpc_weight_env(
            task_file=DEFAULT_TASK_FILE,
            urdf_file=DEFAULT_URDF_FILE,
            ref_file=DEFAULT_REF_FILE,
            headless=headless,
            seed=seed + rank,
            velocity_command=velocity_command,
            reward_fn=reward_fn,
            active_weight_dims=active_weight_dims,
            force_prob=force_prob,
            force_mag_range=force_mag_range,
            vel_cmd_rand_range=vel_cmd_rand_range,
        )
        return TimeLimit(e, max_episode_steps=max_episode_steps)
    return _init


def main():
    headless = os.environ.get("HEADLESS", "1") == "1"
    seed = int(os.environ.get("SEED", "42"))
    n_envs = int(os.environ.get("N_ENVS", "1"))
    # 仅允许 1 / 4 / 8，其它值按 1 处理
    if n_envs not in (1, 4, 8):
        n_envs = 1
    max_episode_steps = 2048

    # ─── 站立抗扰动（加大扰动）配置：纯站立 + 高外力概率/力度 ─────────────────
    load_model_path = os.environ.get("LOAD_MODEL", "ppo_zip/phase1/g1_ppo_residual_v1_phase1_500000_n4.zip").strip()

    phase_active_dims = STANDING_ACTIVE_DIMS
    phase_vel_cmd = np.array([0.0, 0.0, 0.75, 0.0])
    # 加大扰动：默认 85% 概率、45~100 N（Phase 1 为 70%、30~80 N）
    phase_force_prob = float(os.environ.get("FORCE_PROB", "0.85"))
    phase_force_mag = (
        float(os.environ.get("FORCE_MAG_MIN", "45.0")),
        float(os.environ.get("FORCE_MAG_MAX", "100.0")),
    )
    phase_vel_rand = (0.0, 0.0, 0.0)   # 纯站立，无速度随机化
    phase_reward_fn = MpcWeightEnvReward(
        w_height=1.5, w_orientation=0.8, w_velocity=0.0,
        w_survival=0.1, w_action_magnitude=0.005, w_action_smooth=0.01,
        w_gait_symmetry=0.0,
    )
    print(f"[站立抗扰动-加大扰动] active_dims={phase_active_dims}, force_prob={phase_force_prob}, force_mag={phase_force_mag}")

    _env_kwargs = dict(
        velocity_command=phase_vel_cmd,
        reward_fn=phase_reward_fn,
        active_weight_dims=phase_active_dims,
        force_prob=phase_force_prob,
        force_mag_range=phase_force_mag,
        vel_cmd_rand_range=phase_vel_rand,
    )

    if USE_WEIGHT_ENV:
        if n_envs == 1:
            print(f"正在初始化 G1 MPC 权重环境（单 env，仿真同步多速率）...(headless={headless})")
            env = _make_single_env(0, seed, headless, max_episode_steps, **_env_kwargs)()
        else:
            if not headless:
                print("   [提示] N_ENVS>1 时建议 HEADLESS=1，已强制 headless=True 避免多窗口")
                headless = True
            print(f"正在初始化 G1 MPC 权重环境（{n_envs} 个并行 env，SubprocVecEnv）...(headless={headless})")
            env_fns = [_make_single_env(rank, seed, headless, max_episode_steps, **_env_kwargs) for rank in range(n_envs)]
            env = SubprocVecEnv(env_fns)
    else:
        n_envs = 1  # 简化 env 仅支持单 env
        from humanoid_wb_mpc.envs import G1MpcEnv
        print("正在初始化 G1 MPC 强化学习环境（简化）...")
        env = G1MpcEnv(DEFAULT_TASK_FILE, DEFAULT_URDF_FILE, DEFAULT_REF_FILE)
        env = TimeLimit(env, max_episode_steps=max_episode_steps)

    total_steps = int(os.environ.get("TOTAL_TIMESTEPS", 200_000))
    tb_log_dir = os.environ.get("TB_LOG_DIR", "./ppo_g1_logs/phase1_heavy_disturb")

    # PPO 超参（可用环境变量覆盖）
    learning_rate = float(os.environ.get("LEARNING_RATE", "1e-4"))
    n_steps_raw = os.environ.get("N_STEPS", "").strip()
    if n_steps_raw:
        n_steps = int(n_steps_raw)
    else:
        # 多 env 时默认 n_steps = 2048 * n_envs，使每 env 每轮约 2048 步
        n_steps = 2048 * max(1, n_envs)
    batch_size = int(os.environ.get("BATCH_SIZE", "128"))
    n_epochs = int(os.environ.get("N_EPOCHS", "10"))
    gamma = float(os.environ.get("GAMMA", "0.99"))
    clip_range = float(os.environ.get("CLIP_RANGE", "0.1"))
    ent_coef = float(os.environ.get("ENT_COEF", "0.01"))
    target_kl = float(os.environ.get("TARGET_KL", "0.01"))

    # SB3 1.8+ 要求 net_arch 为 dict，与 zip 内保存格式一致，便于续训加载
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))
    _ppo_common = dict(
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
    _load_path = load_model_path.rstrip(".zip") if load_model_path else ""
    if _load_path and os.path.exists(_load_path + ".zip"):
        print(f"   加载模型（站立抗扰动续训）: {_load_path}.zip")
        model = PPO.load(_load_path, env=env, **{k: v for k, v in _ppo_common.items() if k not in ("verbose", "seed")})
        model.verbose = 1
    else:
        if load_model_path:
            print(f"   [警告] LOAD_MODEL={load_model_path} 文件不存在，从头训练")
        model = PPO("MlpPolicy", env, **_ppo_common)

    print(f"🚀 启动训练: G1 站立抗扰动（加大扰动）...")
    print(f"   并行 env: {n_envs} | n_steps: {n_steps} | 总步数: {total_steps:,} | TensorBoard: {tb_log_dir}")
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
    try:
        model.learn(total_timesteps=total_steps, progress_bar=True, callback=callback)
    except EOFError as e:
        print("\n[错误] 并行 env 子进程异常退出（主进程只收到 EOFError）。")
        print("   → 请先用单 env 复现以查看真实异常: N_ENVS=1 ... python3 train_g1_ppo_v1_站立抗扰动导向.py")
        print("   → 若已修改 C++ 绑定，请先重新编译: colcon build --packages-select humanoid_wb_mpc && source install/setup.bash")
        raise SystemExit(1) from e
    save_name = f"g1_ppo_standing_heavy_disturb_{total_steps}" + (f"_n{n_envs}" if n_envs > 1 else "")
    model.save(save_name)
    print(f"✅ 训练模型已保存为 {save_name}.zip")
    tb_abs = os.path.abspath(tb_log_dir)
    print(f"   查看曲线: tensorboard --logdir {tb_abs}")
    print(f"   （或先 cd 到本脚本所在目录再: tensorboard --logdir {tb_log_dir}）")


if __name__ == "__main__":
    main()
