"""
Humanoid MPC Python Package

G1人形机器人MPC控制的Python接口包

主要模块:
- core: 核心工具类 (StateConverter, JointMapper, ActionFormatter)
- sim_interface: MuJoCo仿真环境封装

使用示例:
    from humanoid_wb_mpc import get_mpc_interface, HumanoidEnv

    # 创建MPC接口
    interface = get_mpc_interface(task_file, urdf_file, ref_file)

    # 创建仿真环境
    env = HumanoidEnv(xml_path)
    env.set_mpc_interface(interface)
    env.run()
"""

# 仅立即导入 RL 权重环境，避免加载 core/sim_interface（会 import mujoco）导致无头下 OpenGL 报错
from .envs import (
    G1MpcWeightEnv,
    make_g1_mpc_weight_env,
)

__version__ = "1.0.0"
__author__ = "Humanoid MPC Team"

__all__ = [
    "StateConverter",
    "JointMapper",
    "ActionFormatter",
    "MPC_JOINT_NAMES",
    "WRIST_JOINT_NAMES",
    "ALL_JOINT_NAMES",
    "DEFAULT_PD_GAINS",
    "DEFAULT_MAX_TORQUE",
    "get_mpc_interface",
    "get_controller",
    "MujocoEnv",
    "HumanoidEnv",
    "G1MpcEnv",
    "G1MpcWeightEnv",
    "make_g1_mpc_weight_env",
]


def __getattr__(name):
    """延迟加载 core、sim_interface、G1MpcEnv，避免无头环境下提前 import mujoco 触发 OpenGL 错误。"""
    if name in (
        "StateConverter", "JointMapper", "ActionFormatter",
        "MPC_JOINT_NAMES", "WRIST_JOINT_NAMES", "ALL_JOINT_NAMES",
        "DEFAULT_PD_GAINS", "DEFAULT_MAX_TORQUE",
        "get_mpc_interface", "get_controller",
    ):
        from . import core
        return getattr(core, name)
    if name in ("MujocoEnv", "HumanoidEnv"):
        from . import sim_interface
        return getattr(sim_interface, name)
    if name == "G1MpcEnv":
        from .envs import G1MpcEnv
        return G1MpcEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
