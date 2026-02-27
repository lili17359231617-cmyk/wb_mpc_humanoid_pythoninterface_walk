"""
G1 Humanoid MPC + RL Training Framework

MPC与强化学习结合的完整训练框架
支持:
- 策略梯度训练
- MPC作为教师策略
- 残差动作学习

作者: Humanoid MPC Team
"""

import mujoco
import mujoco.viewer
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import time
import os


# =============================================================================
# 配置
# =============================================================================

@dataclass
class Config:
    """训练配置"""
    # 路径配置
    task_file: str = "/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.info"
    urdf_file: str = "/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf"
    ref_file: str = "/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/robot_models/unitree_g1/g1_wb_mpc/config/command/reference.info"
    xml_file: str = "/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/robot_models/unitree_g1/g1_description/urdf/g1_29dof.xml"

    # 仿真配置
    timestep: float = 0.001
    render: bool = True

    # PD控制配置
    kp: float = 200.0
    kd: float = 5.0
    kp_wrist: float = 20.0
    kd_wrist: float = 0.5
    max_torque: float = 60.0

    # 关节配置
    mpc_joint_names = [
        'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint',
        'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint',
        'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
        'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint', 'left_shoulder_pitch_joint',
        'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint',
        'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint'
    ]

    wrist_joint_names = [
        'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
        'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint'
    ]


# =============================================================================
# 关节映射工具
# =============================================================================

class JointMapper:
    """关节索引映射工具"""

    def __init__(self, model, mpc_joint_names, wrist_joint_names):
        self.model = model
        self.mpc_joint_names = mpc_joint_names
        self.wrist_joint_names = wrist_joint_names

        # 构建索引映射
        self._build_mappings()

    def _build_mappings(self):
        """构建关节索引映射"""
        self.mpc_qpos_idxs = []
        self.mpc_qvel_idxs = []
        self.mpc_ctrl_idxs = []

        self.all_qpos_idxs = []
        self.all_qvel_idxs = []
        self.all_ctrl_idxs = []

        self.all_joint_names = []

        # MPC关节
        for name in self.mpc_joint_names:
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if jnt_id == -1 or act_id == -1:
                print(f"[警告] MPC关节 '{name}' 未找到!")
                continue
            self.mpc_qpos_idxs.append(self.model.jnt_qposadr[jnt_id])
            self.mpc_qvel_idxs.append(self.model.jnt_dofadr[jnt_id])
            self.mpc_ctrl_idxs.append(act_id)
            self.all_qpos_idxs.append(self.model.jnt_qposadr[jnt_id])
            self.all_qvel_idxs.append(self.model.jnt_dofadr[jnt_id])
            self.all_ctrl_idxs.append(act_id)
            self.all_joint_names.append(name)

        # 手腕关节
        for name in self.wrist_joint_names:
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if jnt_id == -1 or act_id == -1:
                continue
            self.all_qpos_idxs.append(self.model.jnt_qposadr[jnt_id])
            self.all_qvel_idxs.append(self.model.jnt_dofadr[jnt_id])
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
    def mpc_joint_dim(self):
        return len(self.mpc_qpos_idxs)

    @property
    def all_joint_dim(self):
        return len(self.all_qpos_idxs)


# =============================================================================
# RL观测空间
# =============================================================================

class ObservationSpace:
    """RL观测空间定义"""

    def __init__(self, joint_mapper: JointMapper):
        self.joint_mapper = joint_mapper

        # 观测维度计算
        # 基座位置: 3
        # 基座欧拉角: 3
        # 基座线速度: 3
        # 基座角速度: 3
        # MPC关节位置: n
        # MPC关节速度: n
        # 目标位置(可选): n
        self.base_pos_dim = 3
        self.base_euler_dim = 3
        self.base_lin_vel_dim = 3
        self.base_ang_vel_dim = 3
        self.joint_pos_dim = joint_mapper.mpc_joint_dim
        self.joint_vel_dim = joint_mapper.mpc_joint_dim

        # 总维度
        self.observation_dim = (
            self.base_pos_dim + self.base_euler_dim +
            self.base_lin_vel_dim + self.base_ang_vel_dim +
            self.joint_pos_dim + self.joint_vel_dim
        )

    def get_observation(self, data) -> np.ndarray:
        """
        获取当前观测

        Args:
            data: MuJoCo数据对象

        Returns:
            观测向量
        """
        obs = np.zeros(self.observation_dim)
        idx = 0

        # 基座位置
        obs[idx:idx+3] = data.qpos[0:3]
        idx += 3

        # 基座欧拉角 (从四元数转换)
        quat_wxyz = data.qpos[3:7]
        quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        from scipy.spatial.transform import Rotation as R
        euler = R.from_quat(quat_xyzw).as_euler('zyx')
        obs[idx:idx+3] = euler
        idx += 3

        # 基座速度
        obs[idx:idx+3] = data.qvel[0:3]
        idx += 3
        obs[idx:idx+3] = data.qvel[3:6]
        idx += 3

        # 关节位置
        for i, qpos_idx in enumerate(self.joint_mapper.mpc_qpos_idxs):
            obs[idx + i] = data.qpos[qpos_idx]
        idx += self.joint_pos_dim

        # 关节速度
        for i, qvel_idx in enumerate(self.joint_mapper.mpc_qvel_idxs):
            obs[idx + i] = data.qvel[qvel_idx]

        return obs


# =============================================================================
# RL动作空间
# =============================================================================

class ActionSpace:
    """RL动作空间定义"""

    def __init__(self, joint_mapper: JointMapper, action_scale: float = 0.1):
        self.joint_mapper = joint_mapper
        self.action_scale = action_scale
        self.action_dim = joint_mapper.mpc_joint_dim
        self.low = -np.ones(self.action_dim)
        self.high = np.ones(self.action_dim)

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        将标准化动作映射到实际控制范围

        Args:
            action: 标准化动作 [-1, 1]

        Returns:
            缩放后的动作
        """
        return action * self.action_scale

    def clip_action(self, action: np.ndarray) -> np.ndarray:
        """
        裁剪动作到有效范围

        Args:
            action: 原始动作

        Returns:
            裁剪后的动作
        """
        return np.clip(action, self.low, self.high)


# =============================================================================
# MPC控制器包装器
# =============================================================================

class MPCController:
    """MPC控制器包装器"""

    def __init__(self, config: Config, joint_mapper: JointMapper):
        self.config = config
        self.joint_mapper = joint_mapper

        # 导入C++ MPC库
        try:
            import humanoid_wb_mpc_py as mpc_lib
            self.mpc_lib = mpc_lib
        except ImportError as e:
            print(f"[警告] 无法导入C++ MPC库: {e}")
            print("将使用简化的PD控制器")
            self.mpc_lib = None

        # 初始化接口
        self.interface = None
        if self.mpc_lib:
            self.interface = self.mpc_lib.WBMpcInterface(
                config.task_file,
                config.urdf_file,
                config.ref_file,
                setup_ocp=True
            )
            print(f"MPC初始化完成 - 状态维度: {self.interface.get_state_dim()}, 输入维度: {self.interface.get_input_dim()}")

    def compute_control(self, obs: np.ndarray, time: float) -> np.ndarray:
        """
        计算MPC控制输出

        Args:
            obs: 当前观测
            time: 当前时间

        Returns:
            MPC输出动作
        """
        if self.interface is None:
            # 返回零动作（使用PD控制）
            return np.zeros(self.joint_mapper.mpc_joint_dim)

        # 构建OCS2格式的状态
        state_dim = self.interface.get_state_dim()
        state = np.zeros(state_dim)

        # 从观测中提取状态
        # 基座位置 [0:3]
        state[0:3] = obs[0:3]

        # 基座欧拉角 [3:6]
        state[3:6] = obs[6:9]

        # 关节位置 [6:6+n]
        state[6:6+self.joint_mapper.mpc_joint_dim] = obs[12:12+self.joint_mapper.mpc_joint_dim]

        # 基座速度 [29:35]
        state[29:32] = obs[9:12]  # 线速度
        state[32:35] = obs[12:15]  # 角速度 (修正索引)

        # 关节速度 [35:35+n]
        state[35:35+self.joint_mapper.mpc_joint_dim] = obs[15+self.joint_mapper.mpc_joint_dim:]

        # 运行MPC
        try:
            mpc_output = self.interface.run_mpc(state, time)
            return mpc_output
        except Exception as e:
            print(f"[MPC错误] {e}")
            return np.zeros(self.joint_mapper.mpc_joint_dim)

    def reset(self):
        """重置MPC状态"""
        if self.interface:
            self.interface.reset()


# =============================================================================
# 奖励函数
# =============================================================================

class RewardFunction:
    """奖励函数定义"""

    def __init__(self, joint_mapper: JointMapper):
        self.joint_mapper = joint_mapper

        # 奖励权重
        self.w_velocity = 0.3  # 前进速度奖励
        self.w_height = 0.5    # 高度保持奖励
        self.w_orientation = 0.3 # 姿态稳定奖励
        self.w_torque = 0.001   # 力矩惩罚
        self.w_joint_acc = 0.001 # 关节加速度惩罚
        self.w_action = 0.01    # 动作变化惩罚

    def compute_reward(self,
                      obs: np.ndarray,
                      action: np.ndarray,
                      next_obs: np.ndarray,
                      info: Dict) -> Tuple[float, Dict]:
        """
        计算奖励

        Args:
            obs: 当前观测
            action: 执行的动作
            next_obs: 下一时刻观测
            info: 附加信息

        Returns:
            (奖励值, 奖励分量字典)
        """
        rewards = {}

        # 1. 前进速度奖励
        velocity = obs[9]  # x方向速度
        rewards['velocity'] = self.w_velocity * max(0, velocity)  # 只奖励前进

        # 2. 高度奖励（保持适当高度）
        height = obs[2]
        target_height = 0.8  # 目标高度
        rewards['height'] = -self.w_height * abs(height - target_height)

        # 3. 姿态奖励（保持直立）
        euler = obs[6:9]
        rewards['orientation'] = -self.w_orientation * (abs(euler[0]) + abs(euler[1]))  # roll和pitch

        # 4. 力矩惩罚
        rewards['torque'] = -self.w_torque * np.sum(np.abs(action))

        # 5. 任务完成奖励
        if info.get('success', False):
            rewards['success'] = 10.0

        # 6. 失败惩罚
        if info.get('fallen', False):
            rewards['fallen'] = -10.0

        total_reward = sum(rewards.values())

        return total_reward, rewards


# =============================================================================
# G1仿真环境
# =============================================================================

class G1Env:
    """G1人形机器人仿真环境"""

    def __init__(self, config: Optional[Config] = None):
        """
        初始化仿真环境

        Args:
            config: 配置对象，若为None则使用默认配置
        """
        self.config = config or Config()

        # 加载模型
        self.model = mujoco.MjModel.from_xml_path(self.config.xml_file)
        self.model.opt.timestep = self.config.timestep
        self.data = mujoco.MjData(self.model)

        # 构建关节映射
        self.joint_mapper = JointMapper(
            self.model,
            self.config.mpc_joint_names,
            self.config.wrist_joint_names
        )

        # 初始化组件
        self.observation_space = ObservationSpace(self.joint_mapper)
        self.action_space = ActionSpace(self.joint_mapper)
        self.reward_function = RewardFunction(self.joint_mapper)
        self.mpc_controller = MPCController(self.config, self.joint_mapper)

        # 渲染器
        self.viewer = None
        self.render_enabled = self.config.render

        # 运行状态
        self.running = False
        self.episode_count = 0
        self.step_count = 0
        self.total_reward = 0.0

        # 重置环境
        self.reset()

    def reset(self) -> np.ndarray:
        """重置环境"""
        mujoco.mj_resetData(self.model, self.data)

        # 设置初始姿态
        initial_pose = np.array([
            -0.05, 0.0, 0.0, 0.1, -0.05, 0.0,  # 左腿
            -0.05, 0.0, 0.0, 0.1, -0.05, 0.0,  # 右腿
            0.0, 0.0, 0.0,                      # 腰部
            0.0, 0.0, 0.0, 0.0,                 # 左臂
            0.0, 0.0, 0.0, 0.0,                 # 右臂
            0.0, 0.0, 0.0,                      # 左手腕
            0.0, 0.0, 0.0,                      # 右手腕
        ])

        # 设置关节位置
        for i, idx in enumerate(self.joint_mapper.all_qpos_idxs):
            if i < len(initial_pose):
                self.data.qpos[idx] = initial_pose[i]

        # 设置关节速度
        for idx in self.joint_mapper.all_qvel_idxs:
            self.data.qvel[idx] = 0.0

        # 前向传播
        mujoco.mj_forward(self.model, self.data)

        # 重置MPC
        self.mpc_controller.reset()

        # 重置计数
        self.step_count = 0
        self.total_reward = 0.0

        # 启动渲染器
        if self.render_enabled and self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行仿真步进

        Args:
            action: RL策略输出的动作

        Returns:
            (next_obs, reward, done, info)
        """
        # 缩放动作
        scaled_action = self.action_space.scale_action(action)

        # 计算控制力矩
        torque = self._compute_torque(scaled_action)

        # 应用控制
        self.data.ctrl[:] = torque

        # 执行仿真
        mujoco.mj_step(self.model, self.data)

        # 同步渲染
        if self.viewer is not None:
            self.viewer.sync()

        # 获取观测
        obs = self._get_obs()
        next_obs = self._get_obs()

        # 计算奖励
        reward, reward_components = self.reward_function.compute_reward(
            obs, scaled_action, next_obs, self._get_info()
        )

        # 更新统计
        self.step_count += 1
        self.total_reward += reward

        # 检查终止条件
        done = self._check_termination()
        info = self._get_info()

        return next_obs, reward, done, info

    def _compute_torque(self, residual_action: np.ndarray) -> np.ndarray:
        """
        计算控制力矩

        Args:
            residual_action: 残差动作

        Returns:
            控制力矩数组
        """
        # 获取MPC输出
        current_obs = self._get_obs()
        mpc_output = self.mpc_controller.compute_control(
            current_obs, self.data.time
        )

        # 组合MPC输出和残差
        torque = np.zeros(self.joint_mapper.all_joint_dim)

        # MPC控制关节
        mpc_joint_dim = self.joint_mapper.mpc_joint_dim

        for i in range(mpc_joint_dim):
            desired_acc = mpc_output[i]

            # PD控制
            current_pos = self.data.qpos[self.joint_mapper.mpc_qpos_idxs[i]]
            current_vel = self.data.qvel[self.joint_mapper.mpc_qvel_idxs[i]]

            # 目标位置 = 0 + 残差
            desired_pos = residual_action[i]

            torque[i] = (self.config.kp * (desired_pos - current_pos) +
                        self.config.kd * (0 - current_vel) +
                        desired_acc * 0.01)  # 积分项

        # 手腕关节
        for i in range(mpc_joint_dim, self.joint_mapper.all_joint_dim):
            current_pos = self.data.qpos[self.joint_mapper.all_qpos_idxs[i]]
            current_vel = self.data.qvel[self.joint_mapper.all_qvel_idxs[i]]

            torque[i] = (self.config.kp_wrist * (0 - current_pos) +
                        self.config.kd_wrist * (0 - current_vel))

        # 限制力矩
        torque = np.clip(torque, -self.config.max_torque, self.config.max_torque)

        return torque

    def _get_obs(self) -> np.ndarray:
        """获取当前观测"""
        return self.observation_space.get_observation(self.data)

    def _get_info(self) -> Dict:
        """获取附加信息"""
        return {
            'time': self.data.time,
            'height': self.data.qpos[2],
            'fallen': self.data.qpos[2] < 0.3,  # 如果高度过低则判定为摔倒
            'velocity': self.data.qvel[0],  # x方向速度
        }

    def _check_termination(self) -> bool:
        """检查是否终止"""
        info = self._get_info()

        # 摔倒终止
        if info['fallen']:
            return True

        # 时间限制
        if self.step_count >= 10000:  # 最大步数
            return True

        return False

    def run(self, max_episodes: int = 10, max_steps_per_episode: int = 10000):
        """
        运行仿真循环

        Args:
            max_episodes: 最大训练轮数
            max_steps_per_episode: 每轮最大步数
        """
        self.running = True

        print("=" * 60)
        print("G1 Humanoid MPC + RL Training")
        print(f"  Observation Dim: {self.observation_space.observation_dim}")
        print(f"  Action Dim: {self.action_space.action_dim}")
        print(f"  MPC Joints: {self.joint_mapper.mpc_joint_dim}")
        print(f"  Total Joints: {self.joint_mapper.all_joint_dim}")
        print("=" * 60)

        for episode in range(max_episodes):
            obs = self.reset()
            episode_reward = 0.0
            episode_steps = 0

            for step in range(max_steps_per_episode):
                # 示例：使用随机策略
                action = np.random.uniform(
                    -1, 1,
                    size=self.action_space.action_dim
                )

                # 执行步进
                next_obs, reward, done, info = self.step(action)
                episode_reward += reward
                episode_steps += 1

                # 打印进度
                if step % 100 == 0:
                    print(f"  Episode {episode+1}/{max_episodes} | "
                          f"Step {step}/{max_steps_per_episode} | "
                          f"Reward: {episode_reward:.2f} | "
                          f"Height: {info['height']:.3f}m | "
                          f"Velocity: {info['velocity']:.3f}m/s")

                if done:
                    break

            print(f"\n  Episode {episode+1} 完成:")
            print(f"    总奖励: {episode_reward:.2f}")
            print(f"    步数: {episode_steps}")
            print(f"    状态: {'摔倒' if info['fallen'] else '正常'}")
            print("-" * 60)

        print("\n训练完成!")

        # 关闭渲染器
        if self.viewer is not None:
            self.viewer.close()

    def close(self):
        """关闭环境"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self.running = False


# =============================================================================
# 策略网络（简化版）
# =============================================================================

class PolicyNetwork:
    """简化的策略网络"""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # 初始化权重
        np.random.seed(42)

        # 双层网络
        self.W1 = np.random.randn(obs_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)

        self.W2 = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.b2 = np.zeros(hidden_dim)

        self.W3 = np.random.randn(hidden_dim, action_dim) * 0.1
        self.b3 = np.zeros(action_dim)

    def forward(self, obs: np.ndarray) -> np.ndarray:
        """
        前向传播

        Args:
            obs: 观测向量

        Returns:
            动作向量
        """
        # ReLU激活
        h1 = np.maximum(0, obs @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)

        # Tanh激活（输出在[-1, 1]范围）
        action = np.tanh(h2 @ self.W3 + self.b3)

        return action

    def update(self, obs: np.ndarray, action: np.ndarray, advantage: np.ndarray,
               learning_rate: float = 0.001):
        """
        策略梯度更新

        Args:
            obs: 观测
            action: 执行的动作
            advantage: 优势函数
            learning_rate: 学习率
        """
        # 简化的策略梯度更新
        # 计算梯度并更新权重

        # 前向传播
        h1 = np.maximum(0, obs @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        action_pred = np.tanh(h2 @ self.W3 + self.b3)

        # 计算策略梯度
        policy_grad = advantage * (action - action_pred)

        # 更新输出层
        self.W3 += learning_rate * np.outer(h2, policy_grad)
        self.b3 += learning_rate * policy_grad

        # 反向传播（简化）
        hidden_grad = policy_grad @ self.W3.T
        hidden_grad[h2 <= 0] = 0  # ReLU梯度

        self.W2 += learning_rate * np.outer(h1, hidden_grad)
        self.b2 += learning_rate * hidden_grad

        hidden_grad = hidden_grad @ self.W2.T
        hidden_grad[h1 <= 0] = 0

        self.W1 += learning_rate * np.outer(obs, hidden_grad)
        self.b1 += learning_rate * hidden_grad


# =============================================================================
# PPO训练器
# =============================================================================

class PPOTrainer:
    """PPO训练器"""

    def __init__(self, env: G1Env, config: Dict = None):
        self.env = env

        # PPO配置
        self.config = config or {
            'learning_rate': 3e-4,
            'clip_epsilon': 0.2,
            'gamma': 0.99,
            'lam': 0.95,
            'update_epochs': 10,
            'minibatch_size': 64,
        }

        # 初始化策略
        self.policy = PolicyNetwork(
            env.observation_space.observation_dim,
            env.action_space.action_dim
        )

    def train(self, num_episodes: int = 1000):
        """
        执行训练

        Args:
            num_episodes: 训练轮数
        """
        print("=" * 60)
        print("开始PPO训练")
        print("=" * 60)

        for episode in range(num_episodes):
            # 收集经验
            obs = self.env.reset()
            episode_reward = 0
            episode_steps = 0

            observations = []
            actions = []
            rewards = []
            log_probs = []
            values = []
            dones = []

            for step in range(10000):
                # 选择动作
                action = self.policy.forward(obs)

                # 执行动作
                next_obs, reward, done, info = self.env.step(action)

                # 存储经验
                observations.append(obs)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)

                episode_reward += reward
                episode_steps += 1

                obs = next_obs

                if done:
                    break

            # 计算回报和优势
            returns = self._compute_returns(rewards, dones)
            advantages = self._compute_advantages(rewards, dones, returns)

            # 更新策略
            self._update_policy(
                observations, actions, returns, advantages
            )

            # 打印进度
            if episode % 10 == 0:
                print(f"Episode {episode+1}/{num_episodes} | "
                      f"Reward: {episode_reward:.2f} | "
                      f"Steps: {episode_steps}")

        print("\n训练完成!")

    def _compute_returns(self, rewards: list, dones: list) -> list:
        """计算回报"""
        returns = []
        R = 0

        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = r + self.config['gamma'] * R
            returns.insert(0, R)

        return returns

    def _compute_advantages(self, rewards: list, dones: list,
                            returns: list) -> list:
        """计算优势函数（简化版）"""
        advantages = []

        for r, R in zip(rewards, returns):
            advantages.append(R - r)

        return advantages

    def _update_policy(self, observations: list, actions: list,
                       returns: list, advantages: list):
        """更新策略"""
        for _ in range(self.config['update_epochs']):
            for i in range(len(observations)):
                obs = observations[i]
                action = actions[i]
                advantage = advantages[i]

                self.policy.update(obs, action, advantage,
                                 self.config['learning_rate'])


# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数"""
    print("=" * 60)
    print("G1 Humanoid MPC + RL Training Framework")
    print("=" * 60)

    # 创建配置
    config = Config()

    # 创建环境
    env = G1Env(config)

    # 运行示例
    print("\n运行示例仿真...")
    env.run(max_episodes=3, max_steps_per_episode=500)

    # 关闭环境
    env.close()

    print("\n示例完成!")
    print("\n要开始训练，请使用:")
    print("  trainer = PPOTrainer(env)")
    print("  trainer.train(num_episodes=1000)")


if __name__ == "__main__":
    main()
