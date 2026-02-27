"""
Python高级封装模块
提供更友好的API，简化MPC控制流程

主要类:
- HumanoidMPC: MPC控制器封装
- HumanoidSimInterface: 仿真器封装
- StateConverter: 状态转换工具
"""

import numpy as np
from typing import Optional, Tuple, Dict, List
from scipy.spatial.transform import Rotation as R

try:
    import humanoid_wb_mpc_py as mpc_lib
except ImportError as e:
    raise ImportError(
        f"无法导入C++ MPC库: {e}\n"
        "请确保已编译Python绑定模块: colcon build --packages-select humanoid_wb_mpc"
    )


# =============================================================================
# 常量定义
# =============================================================================

# 标准关节名称
MPC_JOINT_NAMES = [
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint',
    'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint',
    'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint', 'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint',
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint'
]

WRIST_JOINT_NAMES = [
    'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
    'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint'
]

ALL_JOINT_NAMES = MPC_JOINT_NAMES + WRIST_JOINT_NAMES

# 默认配置
DEFAULT_PD_GAINS = {
    'kp': 200.0,
    'kd': 5.0,
    'kp_wrist': 20.0,
    'kd_wrist': 0.5
}

DEFAULT_MAX_TORQUE = 60.0


# =============================================================================
# 异常类
# =============================================================================

class MPCError(Exception):
    """MPC相关错误"""
    pass


class SimInterfaceError(Exception):
    """仿真接口相关错误"""
    pass


# =============================================================================
# 状态转换工具
# =============================================================================

class StateConverter:
    """
    状态格式转换工具

    提供Mujoco、OCS2 (MPC) 和 NumPy格式之间的状态转换
    """

    # Mujoco状态格式: [x, y, z, qw, qx, qy, qz, qpos...]
    # OCS2状态格式: [pos(3), euler(3), joint_pos(23), vel(3), euler_dot(3), joint_vel(23)]

    def __init__(self, mpc_joint_names: List[str] = MPC_JOINT_NAMES):
        """
        初始化状态转换器

        Args:
            mpc_joint_names: MPC控制的关节名称列表
        """
        self.mpc_joint_names = mpc_joint_names
        self.mpc_joint_dim = len(mpc_joint_names)

    def mujoco_to_ocs2(self,
                       mj_data,
                       mpc_qpos_idxs: np.ndarray,
                       mpc_qvel_idxs: np.ndarray) -> np.ndarray:
        """
        将Mujoco状态转换为OCS2 MPC格式

        Args:
            mj_data: MuJoCo数据对象
            mpc_qpos_idxs: MPC关节在qpos中的索引
            mpc_qvel_idxs: MPC关节在qvel中的索引

        Returns:
            OCS2格式的状态向量
        """
        state_dim = 6 + 3 + self.mpc_joint_dim + 6 + 3 + self.mpc_joint_dim  # 58

        ocs2_state = np.zeros(state_dim)

        # 基座位置 [0:3]
        ocs2_state[0:3] = mj_data.qpos[0:3]

        # 基座四元数转欧拉角 [3:6]
        # Mujoco: [qw, qx, qy, qz], OCS2需要ZYX顺序
        quat_wxyz = mj_data.qpos[3:7]
        quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        euler = R.from_quat(quat_xyzw).as_euler('zyx')
        ocs2_state[3:6] = euler

        # 关节位置 [6:6+mpc_joint_dim]
        for i, idx in enumerate(mpc_qpos_idxs):
            ocs2_state[6 + i] = mj_data.qpos[idx]

        # 基座线速度 [29:32]
        ocs2_state[29:32] = mj_data.qvel[0:3]

        # 基座角速度 [32:35]
        ocs2_state[32:35] = mj_data.qvel[3:6]

        # 关节速度 [35:35+mpc_joint_dim]
        for i, idx in enumerate(mpc_qvel_idxs):
            ocs2_state[35 + i] = mj_data.qvel[idx]

        return ocs2_state

    def ocs2_to_mujoco_target(self,
                               ocs2_action: np.ndarray,
                               current_q: np.ndarray,
                               kp: float = DEFAULT_PD_GAINS['kp'],
                               kd: float = DEFAULT_PD_GAINS['kd'],
                               max_torque: float = DEFAULT_MAX_TORQUE) -> np.ndarray:
        """
        将OCS2 MPC输出转换为Mujoco力矩控制

        Args:
            ocs2_action: OCS2格式的MPC输出（关节加速度）
            current_q: 当前关节位置
            kp: PD位置增益
            kd: PD速度增益
            max_torque: 最大力矩限制

        Returns:
            Mujoco格式的控制力矩数组
        """
        # 简化的PD控制 + 重力补偿
        # 注意：完整实现需要调用mj_inverse计算重力项

        num_joints = len(current_q)
        torque = np.zeros(num_joints)

        # MPC控制的关节使用给定的增益
        for i in range(min(len(ocs2_action), self.mpc_joint_dim)):
            desired_acc = ocs2_action[i]
            error_pos = -current_q[i]  # 假设目标为0
            error_vel = -0  # 假设目标速度为0

            # 二阶系统响应
            torque[i] = kp * error_pos + kd * error_vel + desired_acc * 0.01

        # 限制力矩
        torque = np.clip(torque, -max_torque, max_torque)

        return torque

    def quat_to_euler(self, quat_wxyz: np.ndarray) -> np.ndarray:
        """
        四元数转欧拉角 (ZYX顺序)

        Args:
            quat_wxyz: 四元数 [w, x, y, z]

        Returns:
            欧拉角 [roll, pitch, yaw]
        """
        quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        return R.from_quat(quat_xyzw).as_euler('zyx')

    def euler_to_quat(self, euler: np.ndarray) -> np.ndarray:
        """
        欧拉角转四元数 (ZYX顺序)

        Args:
            euler: 欧拉角 [roll, pitch, yaw]

        Returns:
            四元数 [w, x, y, z]
        """
        return R.from_euler('zyx', euler).as_quat()

    def create_observation(self,
                          mj_data,
                          mpc_qpos_idxs: np.ndarray,
                          mpc_qvel_idxs: np.ndarray,
                          include_history: bool = False,
                          history_length: int = 5) -> np.ndarray:
        """
        创建完整的强化学习观测向量

        Args:
            mj_data: MuJoCo数据对象
            mpc_qpos_idxs: MPC关节位置索引
            mpc_qvel_idxs: MPC关节速度索引
            include_history: 是否包含历史状态
            history_length: 历史长度

        Returns:
            观测向量
        """
        obs = self.mujoco_to_ocs2(mj_data, mpc_qpos_idxs, mpc_qvel_idxs)

        if include_history:
            # 扩展观测向量以包含历史
            full_obs = np.zeros(len(obs) * history_length)
            for i in range(history_length):
                full_obs[i * len(obs):(i + 1) * len(obs)] = obs
            return full_obs

        return obs


# =============================================================================
# 关节映射工具
# =============================================================================

class JointMapper:
    """
    关节索引映射工具

    管理不同关节命名空间之间的映射
    """

    def __init__(self, mj_model, mpc_joint_names: List[str] = MPC_JOINT_NAMES):
        """
        初始化关节映射器

        Args:
            mj_model: MuJoCo模型对象
            mpc_joint_names: MPC控制的关节名称列表
        """
        import mujoco

        self.mj_model = mj_model
        self.mpc_joint_names = mpc_joint_names

        # 构建各种索引映射
        self.mpc_qpos_idxs = []
        self.mpc_qvel_idxs = []
        self.mpc_ctrl_idxs = []

        self.all_qpos_idxs = []
        self.all_qvel_idxs = []
        self.all_ctrl_idxs = []

        self.all_joint_names = []

        # MPC关节
        for name in mpc_joint_names:
            jnt_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            act_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if jnt_id == -1 or act_id == -1:
                print(f"    [警告] MPC关节 '{name}' 在模型中未找到!")
                continue
            self.mpc_qpos_idxs.append(mj_model.jnt_qposadr[jnt_id])
            self.mpc_qvel_idxs.append(mj_model.jnt_dofadr[jnt_id])
            self.mpc_ctrl_idxs.append(act_id)
            self.all_qpos_idxs.append(mj_model.jnt_qposadr[jnt_id])
            self.all_qvel_idxs.append(mj_model.jnt_dofadr[jnt_id])
            self.all_ctrl_idxs.append(act_id)
            self.all_joint_names.append(name)

        # 手腕关节
        for name in WRIST_JOINT_NAMES:
            jnt_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            act_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if jnt_id == -1 or act_id == -1:
                continue
            self.all_qpos_idxs.append(mj_model.jnt_qposadr[jnt_id])
            self.all_qvel_idxs.append(mj_model.jnt_dofadr[jnt_id])
            self.all_ctrl_idxs.append(act_id)
            self.all_joint_names.append(name)

        # 转换为numpy数组
        self.mpc_qpos_idxs = np.array(self.mpc_qpos_idxs)
        self.mpc_qvel_idxs = np.array(self.mpc_qvel_idxs)
        self.mpc_ctrl_idxs = np.array(self.mpc_ctrl_idxs)
        self.all_qpos_idxs = np.array(self.all_qpos_idxs)
        self.all_qvel_idxs = np.array(self.all_qvel_idxs)
        self.all_ctrl_idxs = np.array(self.all_ctrl_idxs)

    @property
    def mpc_joint_dim(self) -> int:
        return len(self.mpc_qpos_idxs)

    @property
    def all_joint_dim(self) -> int:
        return len(self.all_qpos_idxs)


# =============================================================================
# 动作格式化工具
# =============================================================================

class ActionFormatter:
    """
    动作格式化工具

    统一处理不同来源的动作命令
    """

    def __init__(self, joint_mapper: JointMapper):
        """
        初始化动作格式化器

        Args:
            joint_mapper: 关节映射器
        """
        self.joint_mapper = joint_mapper

    def format_mpc_action(self,
                          mpc_output: np.ndarray,
                          current_joint_pos: np.ndarray,
                          current_joint_vel: np.ndarray,
                          pd_gains: Dict = DEFAULT_PD_GAINS,
                          max_torque: float = DEFAULT_MAX_TORQUE) -> np.ndarray:
        """
        格式化MPC输出动作

        Args:
            mpc_output: MPC原始输出
            current_joint_pos: 当前关节位置
            current_joint_vel: 当前关节速度
            pd_gains: PD控制增益字典
            max_torque: 最大力矩限制

        Returns:
            格式化后的力矩命令
        """
        num_joints = self.joint_mapper.all_joint_dim
        torque = np.zeros(num_joints)

        # MPC控制的关节
        mpc_joint_dim = self.joint_mapper.mpc_joint_dim

        for i in range(mpc_joint_dim):
            desired_pos = 0  # 可以从参考轨迹获取
            desired_vel = 0

            # 简单的PD控制
            torque[i] = (pd_gains['kp'] * (desired_pos - current_joint_pos[i]) +
                        pd_gains['kd'] * (desired_vel - current_joint_vel[i]))

        # 手腕关节
        for i in range(mpc_joint_dim, num_joints):
            torque[i] = (pd_gains['kp_wrist'] * (0 - current_joint_pos[i]) +
                        pd_gains['kd_wrist'] * (0 - current_joint_vel[i]))

        # 限制力矩
        torque = np.clip(torque, -max_torque, max_torque)

        return torque

    def format_rl_action(self,
                         rl_action: np.ndarray,
                         mpc_output: np.ndarray,
                         current_joint_pos: np.ndarray,
                         current_joint_vel: np.ndarray,
                         residual_scale: float = 0.1) -> np.ndarray:
        """
        格式化RL残差动作

        Args:
            rl_action: RL策略输出的动作
            mpc_output: MPC原始输出
            current_joint_pos: 当前关节位置
            current_joint_vel: 当前关节速度
            residual_scale: 残差缩放因子

        Returns:
            格式化后的力矩命令
        """
        base_action = self.format_mpc_action(
            mpc_output, current_joint_pos, current_joint_vel
        )

        # 添加残差
        residual = rl_action[:len(base_action)] * residual_scale
        action = base_action + residual

        return action


# =============================================================================
# 简化访问器
# =============================================================================

def get_mpc_interface(task_file: str,
                      urdf_file: str,
                      ref_file: str,
                      setup: bool = True) -> mpc_lib.WBMpcInterface:
    """
    便捷函数：创建MPC接口

    Args:
        task_file: 任务配置文件路径
        urdf_file: URDF模型路径
        ref_file: 参考轨迹文件路径
        setup: 是否立即初始化

    Returns:
        MPC接口对象
    """
    interface = mpc_lib.WBMpcInterface(task_file, urdf_file, ref_file)

    if setup:
        interface.setup_mpc()

    return interface


def get_controller(interface: mpc_lib.WBMpcInterface,
                   frequency: float = 100.0) -> mpc_lib.WBMpcMrtJointController:
    """
    便捷函数：创建MPC控制器

    Args:
        interface: MPC接口对象
        frequency: 控制频率

    Returns:
        MRT控制器对象
    """
    return mpc_lib.WBMpcMrtJointController(interface, frequency)
