"""
G1 人形机器人 MPC 权重 RL 的默认路径与频率常量

路径基于 workspace 根 /wb_humanoid_mpc_ws，可通过 get_g1_config() 或环境变量覆盖。
"""

import os
from typing import Dict, Any

# 工作空间根（便于在安装或不同部署下替换）
_WS_ROOT = os.environ.get("WB_HUMANOID_MPC_WS", "/wb_humanoid_mpc_ws")
_SRC = os.path.join(_WS_ROOT, "src", "wb_humanoid_mpc")
_INSTALL = os.path.join(_WS_ROOT, "install")

# MpcWeightAdjustmentModule 要求的残差维度（与 Q 矩阵状态维一致）
RESIDUAL_WEIGHT_DIM = 58

# 默认配置文件路径
DEFAULT_TASK_FILE = os.path.join(
    _SRC, "robot_models", "unitree_g1", "g1_wb_mpc", "config", "mpc", "task.info"
)
DEFAULT_URDF_FILE = os.path.join(
    _SRC, "robot_models", "unitree_g1", "g1_description", "urdf", "g1_29dof.urdf"
)
DEFAULT_REF_FILE = os.path.join(
    _SRC, "robot_models", "unitree_g1", "g1_wb_mpc", "config", "command", "reference.info"
)
DEFAULT_XML_FILE = os.path.join(
    _SRC, "robot_models", "unitree_g1", "g1_description", "urdf", "g1_29dof.xml"
)
DEFAULT_GAIT_FILE = os.path.join(
    _INSTALL, "humanoid_common_mpc", "share", "humanoid_common_mpc", "config", "command", "gait.info"
)

# 多速率分层默认频率：RL(50Hz) : MPC(100Hz) : Sim(1000Hz)
DEFAULT_RL_FREQUENCY_HZ = 50.0
DEFAULT_MPC_FREQUENCY_HZ = 100.0
DEFAULT_SIM_FREQUENCY_HZ = 1000.0


def get_g1_config(workspace_root: str = None) -> Dict[str, Any]:
    """
    返回 G1 MPC RL 使用的路径与常量字典，便于脚本或环境覆盖。
    若传入 workspace_root，则基于该根重新拼接路径。
    """
    root = workspace_root or _WS_ROOT
    src = os.path.join(root, "src", "wb_humanoid_mpc")
    install = os.path.join(root, "install")
    return {
        "task_file": os.path.join(src, "robot_models", "unitree_g1", "g1_wb_mpc", "config", "mpc", "task.info"),
        "urdf_file": os.path.join(src, "robot_models", "unitree_g1", "g1_description", "urdf", "g1_29dof.urdf"),
        "ref_file": os.path.join(src, "robot_models", "unitree_g1", "g1_wb_mpc", "config", "command", "reference.info"),
        "xml_file": os.path.join(src, "robot_models", "unitree_g1", "g1_description", "urdf", "g1_29dof.xml"),
        "gait_file": os.path.join(install, "humanoid_common_mpc", "share", "humanoid_common_mpc", "config", "command", "gait.info"),
        "residual_weight_dim": RESIDUAL_WEIGHT_DIM,
        "rl_frequency_hz": DEFAULT_RL_FREQUENCY_HZ,
        "mpc_frequency_hz": DEFAULT_MPC_FREQUENCY_HZ,
        "sim_frequency_hz": DEFAULT_SIM_FREQUENCY_HZ,
    }
