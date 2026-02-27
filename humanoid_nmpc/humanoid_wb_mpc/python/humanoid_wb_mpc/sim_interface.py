"""
Mujoco仿真器封装

提供面向对象的Mujoco仿真接口，与MPC控制器配合使用

主要类:
- MujocoEnv: Mujoco仿真环境封装
- HumanoidEnv: 人形机器人专用仿真环境
"""

import numpy as np
from typing import Optional, Dict, Tuple, Callable
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R

from .core import (
    StateConverter, JointMapper, ActionFormatter,
    MPC_JOINT_NAMES, WRIST_JOINT_NAMES,
    DEFAULT_PD_GAINS, DEFAULT_MAX_TORQUE
)


class MujocoEnv:
    """
    MuJoCo仿真环境基类

    提供基本的仿真控制、渲染和数据访问功能
    """

    def __init__(self,
                 xml_path: str,
                 joint_names: list = None,
                 timestep: float = 0.001,
                 render: bool = True):
        """
        初始化仿真环境

        Args:
            xml_path: XML模型文件路径
            joint_names: 受控关节名称列表
            timestep: 仿真时间步长
            render: 是否启用渲染
        """
        self.xml_path = xml_path
        self.joint_names = joint_names or []
        self.timestep = timestep
        self.render_enabled = render

        # 加载模型
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = timestep

        # 初始化数据
        self.data = mujoco.MjData(self.model)

        # 初始化控制器
        self.control_callback = None

        # 运行状态
        self.running = False
        self.viewer = None

    def setup_controller(self, control_func: Callable):
        """
        设置控制回调函数

        Args:
            control_func: 控制函数，接受(data, time)返回控制力矩
        """
        self.control_callback = control_func

    def step(self, torque: np.ndarray):
        """
        执行仿真步进

        Args:
            torque: 控制力矩数组
        """
        # 应用控制
        self.data.ctrl[:] = torque

        # 执行仿真
        mujoco.mj_step(self.model, self.data)

        # 同步渲染
        if self.viewer is not None:
            self.viewer.sync()

    def render(self, mode: str = "human"):
        """
        渲染当前帧

        Args:
            mode: 渲染模式 ("human" 或 "rgb_array")
        """
        if self.render_enabled and mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        elif mode == "rgb_array":
            return mujoco.mj_render(self.model, self.data, width=640, height=480)

    def reset(self):
        """重置仿真到初始状态"""
        mujoco.mj_resetData(self.model, self.data)

    def close(self):
        """关闭仿真环境"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def run(self, max_steps: int = None):
        """
        运行仿真循环

        Args:
            max_steps: 最大步数，None表示无限循环
        """
        self.running = True

        if self.viewer is None and self.render_enabled:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        step_count = 0

        while self.running:
            # 检查退出条件
            if max_steps is not None and step_count >= max_steps:
                break

            # 调用控制回调
            if self.control_callback is not None:
                torque = self.control_callback(self.data, self.data.time)
                self.step(torque)
            else:
                mujoco.mj_step(self.model, self.data)
                if self.viewer is not None:
                    self.viewer.sync()

            step_count += 1

    def stop(self):
        """停止仿真循环"""
        self.running = False

    @property
    def time(self) -> float:
        """获取当前仿真时间"""
        return self.data.time

    @property
    def qpos(self) -> np.ndarray:
        """获取所有关节位置"""
        return self.data.qpos.copy()

    @property
    def qvel(self) -> np.ndarray:
        """获取所有关节速度"""
        return self.data.qvel.copy()

    @property
    def qacc(self) -> np.ndarray:
        """获取所有关节加速度"""
        return self.data.qacc.copy()

    def get_joint_pos(self, joint_indices: np.ndarray) -> np.ndarray:
        """
        获取指定关节的位置

        Args:
            joint_indices: 关节在qpos中的索引

        Returns:
            关节位置数组
        """
        return np.array([self.data.qpos[i] for i in joint_indices])

    def get_joint_vel(self, joint_indices: np.ndarray) -> np.ndarray:
        """
        获取指定关节的速度

        Args:
            joint_indices: 关节在qvel中的索引

        Returns:
            关节速度数组
        """
        return np.array([self.data.qvel[i] for i in joint_indices])


class HumanoidEnv:
    """
    人形机器人仿真环境

    专门用于G1人形机器人的MPC控制仿真
    """

    def __init__(self,
                 xml_path: str,
                 mpc_joint_names: list = MPC_JOINT_NAMES,
                 wrist_joint_names: list = WRIST_JOINT_NAMES,
                 timestep: float = 0.001,
                 render: bool = True,
                 pd_gains: Dict = DEFAULT_PD_GAINS,
                 max_torque: float = DEFAULT_MAX_TORQUE):
        """
        初始化人形机器人仿真环境

        Args:
            xml_path: XML模型文件路径
            mpc_joint_names: MPC控制的关节名称
            wrist_joint_names: 手腕关节名称
            timestep: 仿真时间步长
            render: 是否启用渲染
            pd_gains: PD控制增益
            max_torque: 最大力矩限制
        """
        self.xml_path = xml_path
        self.mpc_joint_names = mpc_joint_names
        self.wrist_joint_names = wrist_joint_names
        self.pd_gains = pd_gains
        self.max_torque = max_torque

        # 加载模型
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = timestep

        # 初始化数据
        self.data = mujoco.MjData(self.model)

        # 构建关节映射
        self.joint_mapper = JointMapper(self.model, mpc_joint_names)
        self.state_converter = StateConverter(mpc_joint_names)
        self.action_formatter = ActionFormatter(self.joint_mapper)

        # 渲染器
        self.viewer = None
        self.render_enabled = render

        # 仿真状态
        self.running = False
        self.mpc_interface = None
        self.controller = None

        # 重置仿真
        self.reset()

    def set_mpc_interface(self, interface, setup: bool = True):
        """
        设置MPC接口

        Args:
            interface: WBMpcInterface对象
            setup: 是否立即初始化MPC
        """
        self.mpc_interface = interface

        if setup:
            interface.setup_mpc()

    def set_controller(self, controller):
        """
        设置MRT控制器

        Args:
            controller: WBMpcMrtJointController对象
        """
        self.controller = controller

    def reset(self, initial_pose: np.ndarray = None):
        """
        重置仿真到初始状态

        Args:
            initial_pose: 初始关节位置，若为None则使用默认姿态
        """
        mujoco.mj_resetData(self.model, self.data)

        # 默认初始姿态
        default_pose = np.array([
            -0.05, 0.0, 0.0, 0.1, -0.05, 0.0,  # 左腿
            -0.05, 0.0, 0.0, 0.1, -0.05, 0.0,  # 右腿
            0.0, 0.0, 0.0,                      # 腰部
            0.0, 0.0, 0.0, 0.0,                 # 左臂
            0.0, 0.0, 0.0, 0.0,                 # 右臂
            0.0, 0.0, 0.0,                      # 左手腕
            0.0, 0.0, 0.0,                      # 右手腕
        ])

        if initial_pose is not None:
            default_pose[:len(initial_pose)] = initial_pose

        # 设置关节位置
        all_qpos_idxs = self.joint_mapper.all_qpos_idxs
        for i, idx in enumerate(all_qpos_idxs):
            if i < len(default_pose):
                self.data.qpos[idx] = default_pose[i]

        # 设置关节速度
        all_qvel_idxs = self.joint_mapper.all_qvel_idxs
        for idx in all_qvel_idxs:
            self.data.qvel[idx] = 0.0

        # 前向传播
        mujoco.mj_forward(self.model, self.data)

    def step(self, torque: np.ndarray = None):
        """
        执行仿真步进

        Args:
            torque: 控制力矩，若为None则使用MPC计算
        """
        # 应用控制
        if torque is not None:
            self.data.ctrl[:] = torque
        elif self.controller is not None:
            # 使用MPC控制器
            try:
                state = self._get_robot_state()
                actions = self.controller.compute_joint_control_action(
                    self.data.time, state, []
                )

                # 将actions转换为力矩
                torque = self._actions_to_torque(actions)
                self.data.ctrl[:] = torque
            except Exception as e:
                print(f"[警告] MPC控制失败: {e}")
                self.data.ctrl.fill(0.0)
        else:
            self.data.ctrl.fill(0.0)

        # 执行仿真
        mujoco.mj_step(self.model, self.data)

        # 同步渲染
        if self.viewer is not None:
            self.viewer.sync()

    def compute_mpc_control(self) -> np.ndarray:
        """
        计算MPC控制力矩

        Returns:
            控制力矩数组
        """
        if self.mpc_interface is None:
            return np.zeros(self.joint_mapper.all_joint_dim)

        # 获取当前状态
        state = self._get_mpc_state()

        # 运行MPC
        mpc_output = self.mpc_interface.run_mpc(state, self.data.time)

        # 获取当前关节状态
        current_pos = self.get_mpc_joint_pos()
        current_vel = self.get_mpc_joint_vel()

        # 格式化动作
        torque = self.action_formatter.format_mpc_action(
            mpc_output, current_pos, current_vel, self.pd_gains, self.max_torque
        )

        return torque

    def _get_robot_state(self):
        """获取C++ RobotState对象"""
        if self.mpc_interface is None:
            return None

        robot_desc = self.mpc_interface.get_robot_description()
        state = mpc_lib.RobotState(robot_desc)

        # 设置基座状态
        state.set_root_position(self.get_base_position())
        state.set_root_rotation_quat(self.get_base_orientation())
        state.set_time(self.data.time)

        # 设置关节状态
        for i, joint_idx in enumerate(self.joint_mapper.all_joint_names):
            try:
                joint_id = robot_desc.get_joint_index(joint_idx)
                state.set_joint_position(joint_id, self.get_joint_pos_by_name(joint_idx))
                state.set_joint_velocity(joint_id, self.get_joint_vel_by_name(joint_idx))
            except:
                pass

        return state

    def _get_mpc_state(self) -> np.ndarray:
        """
        获取MPC格式的状态向量

        Returns:
            OCS2格式的状态向量
        """
        return self.state_converter.mujoco_to_ocs2(
            self.data,
            self.joint_mapper.mpc_qpos_idxs,
            self.joint_mapper.mpc_qvel_idxs
        )

    def _actions_to_torque(self, actions) -> np.ndarray:
        """
        将关节动作转换为力矩

        Args:
            actions: 关节动作列表

        Returns:
            Mujoco格式的控制力矩
        """
        num_joints = self.joint_mapper.all_joint_dim
        torque = np.zeros(num_joints)

        for i in range(min(len(actions), num_joints)):
            action = actions[i]
            torque[i] = action.feed_forward_effort + action.kp * (action.q_des - self.get_all_joint_pos()[i])

        return torque

    def get_base_position(self) -> np.ndarray:
        """
        获取基座位置 [x, y, z]

        Returns:
            基座位置数组
        """
        return self.data.qpos[0:3].copy()

    def get_base_orientation(self) -> np.ndarray:
        """
        获取基座四元数 [w, x, y, z]

        Returns:
            基座四元数
        """
        return self.data.qpos[3:7].copy()

    def get_base_euler(self) -> np.ndarray:
        """
        获取基座欧拉角 [roll, pitch, yaw]

        Returns:
            欧拉角数组
        """
        quat_wxyz = self.data.qpos[3:7]
        quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        return R.from_quat(quat_xyzw).as_euler('zyx')

    def get_base_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取基座速度

        Returns:
            (线速度, 角速度)
        """
        return (self.data.qvel[0:3].copy(), self.data.qvel[3:6].copy())

    def get_mpc_joint_pos(self) -> np.ndarray:
        """
        获取MPC关节位置

        Returns:
            MPC关节位置数组
        """
        return self.joint_mapper.get_joint_pos(self.data)

    def get_mpc_joint_vel(self) -> np.ndarray:
        """
        获取MPC关节速度

        Returns:
            MPC关节速度数组
        """
        return self.joint_mapper.get_joint_vel(self.data)

    def get_all_joint_pos(self) -> np.ndarray:
        """
        获取所有关节位置

        Returns:
            所有关节位置数组
        """
        return np.array([self.data.qpos[i] for i in self.joint_mapper.all_qpos_idxs])

    def get_all_joint_vel(self) -> np.ndarray:
        """
        获取所有关节速度

        Returns:
            所有关节速度数组
        """
        return np.array([self.data.qvel[i] for i in self.joint_mapper.all_qvel_idxs])

    def get_joint_pos_by_name(self, name: str) -> float:
        """
        根据名称获取关节位置

        Args:
            name: 关节名称

        Returns:
            关节位置
        """
        jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return self.data.qpos[self.model.jnt_qposadr[jnt_id]]

    def get_joint_vel_by_name(self, name: str) -> float:
        """
        根据名称获取关节速度

        Args:
            name: 关节名称

        Returns:
            关节速度
        """
        jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return self.data.qvel[self.model.jnt_dofadr[jnt_id]]

    def compute_gravity_compensation(self) -> np.ndarray:
        """
        计算重力补偿力矩

        Returns:
            重力补偿力矩数组
        """
        self.data.qacc[:] = 0
        self.data.xfrc_applied[:] = 0
        mujoco.mj_inverse(self.model, self.data)
        return self.data.qfrc_inverse[6:].copy()

    def launch_viewer(self):
        """启动可视化"""
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def close_viewer(self):
        """关闭可视化"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def run(self, max_steps: int = None, use_mpc: bool = True):
        """
        运行仿真循环

        Args:
            max_steps: 最大步数，None表示无限循环
            use_mpc: 是否使用MPC控制
        """
        self.running = True

        # 启动渲染器
        if self.render_enabled and self.viewer is None:
            self.launch_viewer()

        step_count = 0

        print("=" * 60)
        print("G1 Humanoid Simulation Started")
        print(f"  MPC Joints: {self.joint_mapper.mpc_joint_dim}")
        print(f"  Total Joints: {self.joint_mapper.all_joint_dim}")
        print("=" * 60)

        while self.running:
            if max_steps is not None and step_count >= max_steps:
                break

            # 计算控制
            if use_mpc:
                torque = self.compute_mpc_control()
            else:
                torque = np.zeros(self.joint_mapper.all_joint_dim)

            # 执行仿真
            self.step(torque)

            # 调试输出
            if step_count % 100 == 0:
                base_pos = self.get_base_position()
                base_euler = self.get_base_euler()
                print(f"  Step: {step_count:5d} | "
                      f"Time: {self.data.time:.2f}s | "
                      f"Height: {base_pos[2]:.3f}m | "
                      f"Att: [{base_euler[0]:.2f}, {base_euler[1]:.2f}, {base_euler[2]:.2f}]")

            step_count += 1

        print("=" * 60)
        print("Simulation Ended")
        print("=" * 60)

    def stop(self):
        """停止仿真"""
        self.running = False

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.close_viewer()
        return False
