"""
G1 MPC 权重 PPO 训练脚本

使用包内配置与环境：默认使用 G1MpcWeightEnv（仿真同步多速率），
也可切换为 G1MpcEnv（简化 MuJoCo 步进）。路径从 humanoid_wb_mpc.config 读取。
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
from gymnasium.wrappers import TimeLimit
from humanoid_wb_mpc.config import (
    DEFAULT_TASK_FILE,
    DEFAULT_URDF_FILE,
    DEFAULT_REF_FILE,
)
from humanoid_wb_mpc import make_g1_mpc_weight_env

# 是否使用完整仿真环境（G1MpcWeightEnv）；False 则使用简化 G1MpcEnv（会导入 mujoco，需 GL）
USE_WEIGHT_ENV = True


def main():
    headless = os.environ.get("HEADLESS", "1") == "1"

    if USE_WEIGHT_ENV:
        print(f"正在初始化 G1 MPC 权重环境（仿真同步多速率）...(headless={headless})")
        env = make_g1_mpc_weight_env(
            task_file=DEFAULT_TASK_FILE,
            urdf_file=DEFAULT_URDF_FILE,
            ref_file=DEFAULT_REF_FILE,
            headless=headless,
            seed=42,
        )
    else:
        from humanoid_wb_mpc.envs import G1MpcEnv
        print("正在初始化 G1 MPC 强化学习环境（简化）...")
        env = G1MpcEnv(DEFAULT_TASK_FILE, DEFAULT_URDF_FILE, DEFAULT_REF_FILE)

    max_episode_steps = 2048
    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    total_steps = int(os.environ.get("TOTAL_TIMESTEPS", 200_000))
    tb_log_dir = os.environ.get("TB_LOG_DIR", "./ppo_g1_logs/phase1_short")

    policy_kwargs = dict(net_arch=[dict(pi=[256, 256], qf=[256, 256])])
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=tb_log_dir,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.1,
        ent_coef=0.01,
        target_kl=0.01,
        policy_kwargs=policy_kwargs,
        device="auto",
    )

    # 第一阶段短训练试跑：1e5～2e5 steps（单 env），用于观察 reward / episode length 曲线
    print("🚀 启动训练: G1 步态残差优化（Phase1 短训练）...")
    print(f"   总步数: {total_steps:,} | TensorBoard: {tb_log_dir}")
    model.learn(total_timesteps=total_steps, progress_bar=True)
    save_name = f"g1_ppo_residual_v1_phase1_{total_steps}"
    model.save(save_name)
    print(f"✅ 训练模型已保存为 {save_name}.zip")
    tb_abs = os.path.abspath(tb_log_dir)
    print(f"   查看曲线: tensorboard --logdir {tb_abs}")
    print(f"   （或先 cd 到本脚本所在目录再: tensorboard --logdir {tb_log_dir}）")


if __name__ == "__main__":
    main()
