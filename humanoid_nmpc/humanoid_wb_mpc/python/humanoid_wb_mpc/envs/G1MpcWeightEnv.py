"""
G1 人形机器人「RL 调节 MPC 权重」训练环境

仿真同步、多速率分层：
- 不启动 MPC 后台线程；每 RL 步先 set_residual_weights(action)，再按 MPC 频率执行「MPC 求解 + 仿真步」。
- 多速率默认 RL(50Hz) : MPC(100Hz) : Sim(1000Hz)。
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Optional, Dict, Any
from scipy.spatial.transform import Rotation as R

# 使用包内 bootstrap 与配置（在创建环境前需已 import humanoid_wb_mpc.bootstrap）
from humanoid_wb_mpc.config import (
    RESIDUAL_WEIGHT_DIM,
    DEFAULT_TASK_FILE,
    DEFAULT_URDF_FILE,
    DEFAULT_REF_FILE,
    DEFAULT_XML_FILE,
    DEFAULT_GAIT_FILE,
    DEFAULT_RL_FREQUENCY_HZ,
    DEFAULT_MPC_FREQUENCY_HZ,
    DEFAULT_SIM_FREQUENCY_HZ,
)
from humanoid_wb_mpc.spaces import MpcResidualActionSpace
from humanoid_wb_mpc.utils.reward_handler import MpcWeightEnvReward


def _get_mpc_py():
    from humanoid_wb_mpc import bootstrap
    return bootstrap.ensure_mpc_py()


def robot_state_to_observation(controller, robot_state) -> np.ndarray:
    """
    从当前 RobotState 得到观测向量（与 MPC 状态同构）。
    状态维数由 interface.get_state_dim() 给出，通常为 58。
    """
    mpc_state = controller.get_mpc_state_from_robot_state(robot_state)
    return np.array(mpc_state, dtype=np.float32)


def get_observation_dim(interface) -> int:
    return int(interface.get_state_dim())


class G1MpcWeightEnv(gym.Env):
    """
    仿真同步、多速率分层的 RL 环境（Gymnasium 兼容，供 SB3 等使用）：
    - 每 RL 步先 set_residual_weights(action)，再在当步内按 MPC 频率执行多次「MPC 求解 + 若干仿真步」。
    - 多速率：RL(50Hz) : MPC(100Hz) : Sim(1000Hz)。
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        task_file: str = DEFAULT_TASK_FILE,
        urdf_file: str = DEFAULT_URDF_FILE,
        ref_file: str = DEFAULT_REF_FILE,
        xml_file: str = DEFAULT_XML_FILE,
        gait_file: str = DEFAULT_GAIT_FILE,
        rl_frequency_hz: float = DEFAULT_RL_FREQUENCY_HZ,
        mpc_frequency_hz: float = DEFAULT_MPC_FREQUENCY_HZ,
        sim_frequency_hz: float = DEFAULT_SIM_FREQUENCY_HZ,
        headless: bool = True,
        velocity_command: Optional[np.ndarray] = None,
        reward_fn: Optional[MpcWeightEnvReward] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        mpc_py = _get_mpc_py()

        self._rl_frequency_hz = rl_frequency_hz
        self._mpc_frequency_hz = mpc_frequency_hz
        self._sim_frequency_hz = sim_frequency_hz
        self._rl_dt = 1.0 / rl_frequency_hz
        self._mpc_dt = 1.0 / mpc_frequency_hz
        self._sim_dt = 1.0 / sim_frequency_hz
        self._mpc_steps_per_rl = int(round(self._rl_dt / self._mpc_dt))
        self._sim_steps_per_mpc = int(round(self._mpc_dt / self._sim_dt))
        self._sim_steps_per_rl = self._mpc_steps_per_rl * self._sim_steps_per_mpc
        assert self._mpc_steps_per_rl >= 1 and self._sim_steps_per_mpc >= 1, (
            "RL/MPC/Sim 频率需满足 rl_dt >= mpc_dt >= sim_dt"
        )

        self._headless = headless
        self._urdf_file = urdf_file
        self._velocity_command = velocity_command if velocity_command is not None else np.array([0.0, 0.0, 0.75, 0.0])
        self._reward_fn = reward_fn or MpcWeightEnvReward()
        self._seed = seed
        if seed is not None:
            np.random.seed(seed)

        self._interface = mpc_py.WBMpcInterface(task_file, urdf_file, ref_file, setup_ocp=True)
        self._interface.setup_mpc()
        self._state_dim = self._interface.get_state_dim()
        self._weight_module = self._interface.get_weight_adjustment_module()

        self._proc_mgr = mpc_py.ProceduralMpcMotionManager(gait_file, ref_file, self._interface)
        init_mpc_state = np.array(self._interface.get_initial_state(), dtype=float)
        initial_base_height = float(init_mpc_state[2])
        cmd = np.array([
            self._velocity_command[0], self._velocity_command[1],
            initial_base_height, self._velocity_command[3]
        ])
        self._proc_mgr.set_velocity_command(cmd)

        model_settings = self._interface.get_model_settings()
        mpc_joint_dim = model_settings.mpc_joint_dim
        robot_description = mpc_py.RobotDescription(urdf_file)
        init_robot_state = mpc_py.RobotState(robot_description, 2)
        base_pos = init_mpc_state[:3]
        base_euler = init_mpc_state[3:6]
        mpc_joint_angles = init_mpc_state[6 : 6 + mpc_joint_dim]
        quat_xyzw = R.from_euler("zyx", base_euler, degrees=False).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        init_robot_state.set_root_position(base_pos)
        init_robot_state.set_root_rotation_quat(quat_wxyz)
        mpc_joint_names = model_settings.mpc_model_joint_names
        for i, jname in enumerate(mpc_joint_names):
            try:
                jidx = robot_description.get_joint_index(jname)
                init_robot_state.set_joint_position(jidx, mpc_joint_angles[i])
                init_robot_state.set_joint_velocity(jidx, 0.0)
            except Exception:
                pass
        init_robot_state.set_root_linear_velocity(np.zeros(3))
        init_robot_state.set_root_angular_velocity(np.zeros(3))

        sim_cfg = mpc_py.MujocoSimConfig()
        sim_cfg.scene_path = xml_file
        sim_cfg.dt = self._sim_dt
        sim_cfg.render_frequency_hz = 60.0
        sim_cfg.headless = headless
        sim_cfg.verbose = False
        sim_cfg.set_init_state(init_robot_state)

        self._sim = mpc_py.MujocoSimInterface(sim_cfg, urdf_file)
        self._sim.init_sim()

        self._controller = mpc_py.WBMpcMrtJointController(self._interface, mpc_frequency_hz)
        self._joint_actions = self._sim.get_robot_joint_action()

        self._obs_dim = get_observation_dim(self._interface)
        self._action_clipper = MpcResidualActionSpace()
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(RESIDUAL_WEIGHT_DIM,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32
        )

        self._prev_action: Optional[np.ndarray] = None
        self._sim_time = 0.0
        self._step_count = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            np.random.seed(seed)
        options = options or {}
        init_mpc_state = self._interface.get_initial_state()
        model_settings = self._interface.get_model_settings()
        mpc_joint_dim = model_settings.mpc_joint_dim
        base_pos = init_mpc_state[:3]
        base_euler = init_mpc_state[3:6]
        mpc_joint_angles = init_mpc_state[6 : 6 + mpc_joint_dim]
        quat_xyzw = R.from_euler("zyx", base_euler, degrees=False).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        robot_description = _get_mpc_py().RobotDescription(self._urdf_file)
        init_robot_state = _get_mpc_py().RobotState(robot_description, 2)
        init_robot_state.set_root_position(base_pos)
        init_robot_state.set_root_rotation_quat(quat_wxyz)
        mpc_joint_names = model_settings.mpc_model_joint_names
        for i, jname in enumerate(mpc_joint_names):
            try:
                jidx = robot_description.get_joint_index(jname)
                init_robot_state.set_joint_position(jidx, mpc_joint_angles[i])
                init_robot_state.set_joint_velocity(jidx, 0.0)
            except Exception:
                pass
        init_robot_state.set_root_linear_velocity(np.zeros(3))
        init_robot_state.set_root_angular_velocity(np.zeros(3))
        try:
            self._sim.set_robot_state(init_robot_state)
        except Exception:
            self._sim.reset()

        self._interface.reset()
        self._proc_mgr.set_velocity_command(
            np.array([
                self._velocity_command[0], self._velocity_command[1],
                float(base_pos[2]), self._velocity_command[3]
            ])
        )
        self._prev_action = None
        self._sim_time = 0.0
        self._step_count = 0

        self._sim.update_interface_state_from_robot()
        robot_state = self._sim.get_robot_state()
        obs = robot_state_to_observation(self._controller, robot_state)
        info = {"sim_time": self._sim_time, "step": self._step_count}
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.asarray(action, dtype=np.float64)
        if action.size != RESIDUAL_WEIGHT_DIM:
            action = np.resize(action, RESIDUAL_WEIGHT_DIM)
        action = self._action_clipper.clip(action)

        self._weight_module.set_residual_weights(action.tolist())

        for _ in range(self._mpc_steps_per_rl):
            self._sim.update_interface_state_from_robot()
            robot_state = self._sim.get_robot_state()
            self._controller.compute_joint_control_action_sync(
                robot_state.get_time(),
                robot_state,
                self._joint_actions,
            )
            for _ in range(self._sim_steps_per_mpc):
                self._sim.apply_joint_action()
                self._sim.simulation_step()

        self._sim_time = self._sim_time + self._rl_dt
        self._step_count += 1

        self._sim.update_interface_state_from_robot()
        next_robot_state = self._sim.get_robot_state()
        next_obs = robot_state_to_observation(self._controller, next_robot_state)

        height = next_robot_state.get_root_position()[2]
        terminated = height < self._reward_fn.fall_height_threshold
        truncated = False
        info = {
            "sim_time": self._sim_time,
            "step": self._step_count,
            "height": height,
            "fallen": terminated,
        }

        reward, reward_components = self._reward_fn.compute(next_obs, action, self._prev_action, info)
        info["reward_components"] = reward_components
        self._prev_action = action.copy()

        return next_obs, float(reward), terminated, truncated, info

    @property
    def unwrapped(self):
        return self


def make_g1_mpc_weight_env(
    task_file: str = DEFAULT_TASK_FILE,
    urdf_file: str = DEFAULT_URDF_FILE,
    ref_file: str = DEFAULT_REF_FILE,
    xml_file: str = DEFAULT_XML_FILE,
    gait_file: str = DEFAULT_GAIT_FILE,
    rl_frequency_hz: float = DEFAULT_RL_FREQUENCY_HZ,
    mpc_frequency_hz: float = DEFAULT_MPC_FREQUENCY_HZ,
    sim_frequency_hz: float = DEFAULT_SIM_FREQUENCY_HZ,
    headless: bool = True,
    velocity_command: Optional[np.ndarray] = None,
    reward_fn: Optional[MpcWeightEnvReward] = None,
    seed: Optional[int] = None,
) -> G1MpcWeightEnv:
    """构造 G1 MPC 权重调节 RL 环境。多速率默认 RL(50Hz) : MPC(100Hz) : Sim(1000Hz)。"""
    return G1MpcWeightEnv(
        task_file=task_file,
        urdf_file=urdf_file,
        ref_file=ref_file,
        xml_file=xml_file,
        gait_file=gait_file,
        rl_frequency_hz=rl_frequency_hz,
        mpc_frequency_hz=mpc_frequency_hz,
        sim_frequency_hz=sim_frequency_hz,
        headless=headless,
        velocity_command=velocity_command,
        reward_fn=reward_fn,
        seed=seed,
    )
