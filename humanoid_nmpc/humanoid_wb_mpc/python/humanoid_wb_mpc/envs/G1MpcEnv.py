import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
import humanoid_wb_mpc_py

class G1MpcEnv(gym.Env):
    def __init__(self, task_file, urdf_file, ref_file):
        super().__init__()
        # 1. 初始化底层 C++ 接口
        self.interface = humanoid_wb_mpc_py.WBMpcInterface(task_file, urdf_file, ref_file)
        self.weight_module = humanoid_wb_mpc_py.MpcWeightAdjustmentModule(self.interface)

        # 2. 加载 MuJoCo 模型与数据 (这是你之前代码中缺失的)
        # 建议直接从 urdf 加载或指定对应的 xml
        self.model = mujoco.MjModel.from_xml_path(urdf_file.replace(".urdf", ".xml")) # 示例路径处理
        self.data = mujoco.MjData(self.model)

        # 3. 定义空间
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(58,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(58,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        # Gymnasium 强制要求处理 seed
        super().reset(seed=seed)

        # 重置物理引擎状态
        mujoco.mj_resetData(self.model, self.data)

        # 获取初始观测
        obs = self._get_obs(self.data)
        info = {}

        # 修复：必须返回元组 (obs, info)
        return obs, info

    def _get_obs(self, mj_data):
        # 基座与关节数据提取
        base_pos = mj_data.qpos[0:3]
        quat = mj_data.qpos[3:7]
        euler_zyx = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler('zyx')
        joint_pos = mj_data.qpos[7:30] # 23个关节

        base_lin_vel = mj_data.qvel[0:3]
        base_ang_vel = mj_data.qvel[3:6][[2, 1, 0]] # 调整为 z, y, x 顺序
        joint_vel = mj_data.qvel[6:29]

        obs = np.concatenate([
            base_pos, euler_zyx, joint_pos,
            base_lin_vel, base_ang_vel, joint_vel
        ])
        return obs.astype(np.float32)

    def step(self, action):
        # 1. 将 58 维残差注入 MPC 权重矩阵
        self.weight_module.set_residual_weights(action.tolist())

        # 2. 执行物理步进 (例如运行 10 次子步以匹配 MPC 频率)
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        # 3. 获取反馈
        obs = self._get_obs(self.data)
        reward = self._compute_reward(obs, action)

        # 4. 判定终止条件 (例如基座高度低于 0.45m 则认为跌倒)
        terminated = bool(obs[2] < 0.35)
        truncated = False
        info = {}

        # 修复：返回 5 个值符合 Gymnasium 标准
        return obs, reward, terminated, truncated, info

    def _compute_reward(self, obs, action):
        # 简单的示例奖励：鼓励保持高度且惩罚过大的残差动作
        height_reward = np.exp(-10.0 * (obs[2] - 0.8)**2) # 假设 0.8m 是站立高度
        action_penalty = -0.01 * np.sum(np.square(action))
        return height_reward + action_penalty