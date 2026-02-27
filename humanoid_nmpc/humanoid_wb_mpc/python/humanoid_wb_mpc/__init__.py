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

from .core import (
    StateConverter,
    JointMapper,
    ActionFormatter,
    MPC_JOINT_NAMES,
    WRIST_JOINT_NAMES,
    ALL_JOINT_NAMES,
    DEFAULT_PD_GAINS,
    DEFAULT_MAX_TORQUE,
    get_mpc_interface,
    get_controller
)

from .sim_interface import (
    MujocoEnv,
    HumanoidEnv
)

__version__ = "1.0.0"
__author__ = "Humanoid MPC Team"

__all__ = [
    # 核心工具
    "StateConverter",
    "JointMapper",
    "ActionFormatter",
    # 常量
    "MPC_JOINT_NAMES",
    "WRIST_JOINT_NAMES",
    "ALL_JOINT_NAMES",
    "DEFAULT_PD_GAINS",
    "DEFAULT_MAX_TORQUE",
    # 便捷函数
    "get_mpc_interface",
    "get_controller",
    # 仿真环境
    "MujocoEnv",
    "HumanoidEnv",
]
