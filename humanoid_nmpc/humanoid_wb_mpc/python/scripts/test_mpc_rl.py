#!/usr/bin/env python3
"""
G1 RL 调节 MPC 权重 — 仿真同步多速率环境测试

使用包内环境与配置，运行简短随机策略测试。
必须在导入 humanoid_wb_mpc 前将 C++ 扩展路径加入 sys.path，否则包内 core 会先导入 humanoid_wb_mpc_py 导致失败。
"""

import os
import sys

# 1) 将 python 包根目录加入路径
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# 2) 在导入 humanoid_wb_mpc 之前加入 C++ 扩展路径（否则 __init__.py -> core 会先 import humanoid_wb_mpc_py 报错）
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

# 3) MuJoCo 渲染后端（在 import 任何使用 mujoco 的模块前设置）
if "MUJOCO_GL" not in os.environ:
    if os.environ.get("HEADLESS", "0") == "1" or not os.environ.get("DISPLAY"):
        os.environ["MUJOCO_GL"] = "egl"
    else:
        os.environ["MUJOCO_GL"] = "glfw"

# 4) 再导入包（bootstrap 内会再次加路径、预加载 glfw/GLEW）
import humanoid_wb_mpc.bootstrap  # noqa: F401
from humanoid_wb_mpc import make_g1_mpc_weight_env


def main():
    print("=" * 60)
    print("G1 RL 调节 MPC 权重 — 仿真同步多速率环境测试")
    print("=" * 60)

    headless = os.environ.get("HEADLESS", "1") == "1"
    env = make_g1_mpc_weight_env(headless=headless, seed=42)
    obs, info = env.reset(seed=42)
    print(f"观测维度: {obs.shape}, 动作维度: {env.action_space.dim}")

    total_reward = 0.0
    max_steps = 200
    t = 0
    for t in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if (t + 1) % 50 == 0:
            print(f"  step {t+1}  height={info.get('height', 0):.3f}  reward={reward:.4f}  total={total_reward:.2f}")
        if terminated or truncated:
            print(f"  终止 @ step {t+1} (terminated={terminated}, truncated={truncated})")
            break

    print("-" * 60)
    print(f"总步数: {min(max_steps, t+1)}, 总奖励: {total_reward:.2f}")
    print("完成")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
