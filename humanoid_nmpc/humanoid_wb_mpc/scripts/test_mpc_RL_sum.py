#!/usr/bin/env python3
"""
G1 人形机器人「RL 调节 MPC 权重」训练环境

将原实时异步 MPC-MRT 测试脚本改造为仿真同步、多速率分层的 RL 环境：
1. 核心改造：从「实时异步」转为「仿真同步」—— 不启动 MPC 后台线程，每步先注入 RL 残差权重，
   再调用 compute_joint_control_action_sync 在当线程内执行一次 MPC 求解 + 关节控制，最后仿真步进。
2. 多速率分层：RL(50Hz) : MPC(100Hz) : Sim(1000Hz)，即每 RL 步内执行 2 次 MPC、每次 MPC 后执行 10 次仿真步。
3. 动作空间：58 维残差 a，对应 MpcWeightAdjustmentModule.set_residual_weights(a)，映射为 Q_i = Q_base_i * exp(a_i)（与 Q 矩阵状态维一致）。
4. 观测空间：基于当前 RobotState 得到的 MPC 状态（基座位姿、关节位姿、速度等）。
5. 奖励函数：跟踪速度/高度、姿态稳定、权重平滑等。

使用方式：
- 作为 Gymnasium 风格环境：env = make_g1_mpc_weight_env(); obs, info = env.reset(); obs, r, term, trunc, info = env.step(action)
- 或直接运行本脚本进行简短随机策略测试。
"""

import sys
import os
import numpy as np
from typing import Tuple, Optional, Dict, Any

# ==========================
# MuJoCo 渲染后端设置
# ==========================
if "MUJOCO_GL" not in os.environ:
    has_display = os.environ.get("DISPLAY") is not None
    is_headless = os.environ.get("HEADLESS", "0") == "1"
    if is_headless or not has_display:
        os.environ["MUJOCO_GL"] = "egl"
    else:
        os.environ["MUJOCO_GL"] = "glfw"

# ==========================
# Python C++ 扩展导入
# ==========================
_mpc_lib = "/wb_humanoid_mpc_ws/install/humanoid_wb_mpc/lib"
if os.path.isdir(_mpc_lib) and _mpc_lib not in sys.path:
    sys.path.insert(0, _mpc_lib)

try:
    import ctypes
    _r = ctypes.RTLD_GLOBAL
    try:
        ctypes.CDLL("libglfw.so.3", mode=_r)
    except OSError:
        try:
            ctypes.CDLL("libglfw.so", mode=_r)
        except OSError:
            pass
    try:
        ctypes.CDLL("libGLEW.so.2.2", mode=_r)
    except OSError:
        try:
            ctypes.CDLL("libGLEW.so", mode=_r)
        except OSError:
            pass
except Exception:
    pass

try:
    import humanoid_wb_mpc_py as mpc_py
except ImportError as e:
    print(f"[错误] 无法导入 humanoid_wb_mpc_py: {e}")
    sys.exit(1)


# =============================================================================
# 常量与路径
# =============================================================================

RESIDUAL_WEIGHT_DIM = 58  # MpcWeightAdjustmentModule 要求的残差维度（与 Q 矩阵状态维一致）

DEFAULT_TASK_FILE = (
    "/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.info"
)
DEFAULT_URDF_FILE = (
    "/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf"
)
DEFAULT_REF_FILE = (
    "/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/robot_models/unitree_g1/g1_wb_mpc/config/command/reference.info"
)
DEFAULT_XML_FILE = (
    "/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/robot_models/unitree_g1/g1_description/urdf/g1_29dof.xml"
)
DEFAULT_GAIT_FILE = (
    "/wb_humanoid_mpc_ws/install/humanoid_common_mpc/share/humanoid_common_mpc/config/command/gait.info"
)


# =============================================================================
# 1. 动作空间（RL 输出 MpcWeightAdjustmentModule 的残差权重）
# =============================================================================

class MpcResidualActionSpace:
    """
    RL 动作空间：58 维残差权重（与 MPC 状态/Q 维一致）。
    在 C++ 侧映射为 Q_i = Q_base_i * exp(a_i)，故 a 通常取较小范围（如 [-1, 1]）即可。
    """

    def __init__(self, low: float = -1.0, high: float = 1.0):
        self.dim = RESIDUAL_WEIGHT_DIM
        self.low = np.full(self.dim, low, dtype=np.float32)
        self.high = np.full(self.dim, high, dtype=np.float32)

    def sample(self) -> np.ndarray:
        return np.random.uniform(self.low, self.high).astype(np.float32)

    def clip(self, action: np.ndarray) -> np.ndarray:
        return np.clip(action, self.low, self.high)


# =============================================================================
# 2. 观测空间（基于 RobotState / MPC 状态）
# =============================================================================

def robot_state_to_observation(controller: "mpc_py.WBMpcMrtJointController", robot_state) -> np.ndarray:
    """
    从当前 RobotState 得到观测向量（与 MPC 状态同构，便于策略输入）。
    状态维数由 interface.get_state_dim() 给出，通常为 58：
    [base_pos(3), base_euler(3), joint_pos(23), base_vel(3), euler_dot(3), joint_vel(23)]。
    """
    mpc_state = controller.get_mpc_state_from_robot_state(robot_state)
    return np.array(mpc_state, dtype=np.float32)


def get_observation_dim(interface) -> int:
    return int(interface.get_state_dim())


# =============================================================================
# 3. 奖励函数
# =============================================================================

class MpcWeightEnvReward:
    """
    奖励函数：鼓励跟踪指令、保持高度与姿态、惩罚跌倒与过大/剧烈变化的权重。
    """

    def __init__(
        self,
        target_height: float = 0.75,
        target_vx: float = 0.0,
        target_vy: float = 0.0,
        target_vyaw: float = 0.0,
        w_height: float = 1.0,
        w_velocity: float = 0.5,
        w_orientation: float = 0.5,
        w_action_magnitude: float = 0.01,
        w_action_smooth: float = 0.02,
        fall_height_threshold: float = 0.35,
        max_roll_pitch: float = 0.5,
    ):
        self.target_height = target_height
        self.target_vx = target_vx
        self.target_vy = target_vy
        self.target_vyaw = target_vyaw
        self.w_height = w_height
        self.w_velocity = w_velocity
        self.w_orientation = w_orientation
        self.w_action_magnitude = w_action_magnitude
        self.w_action_smooth = w_action_smooth
        self.fall_height_threshold = fall_height_threshold
        self.max_roll_pitch = max_roll_pitch

    def compute(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        prev_action: Optional[np.ndarray],
        info: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float]]:
        """
        obs 与 MPC 状态同构: [base_pos(3), base_euler(3), joint_pos(23), base_vel(3), euler_dot(3), joint_vel(23)]
        即 base_pos=obs[0:3], base_euler=obs[3:6], base_vel=obs[29:32], base_angular≈obs[32:35] 等（需与 state_dim 布局一致）。
        """
        components = {}
        # 高度：obs[2] 为 base z
        height = float(obs[2])
        components["height"] = -self.w_height * abs(height - self.target_height)
        # 速度跟踪：若 state 中线速度在 [29:32]
        state_dim = obs.size
        if state_dim >= 32:
            vx, vy = float(obs[29]), float(obs[30])
            components["velocity"] = (
                -self.w_velocity
                * (abs(vx - self.target_vx) + abs(vy - self.target_vy))
            )
        else:
            components["velocity"] = 0.0
        # 姿态：obs[3:5] 为 roll, pitch
        roll, pitch = float(obs[3]), float(obs[4])
        components["orientation"] = -self.w_orientation * (
            min(abs(roll), self.max_roll_pitch) + min(abs(pitch), self.max_roll_pitch)
        )
        # 动作幅值惩罚（避免权重过大）
        components["action_magnitude"] = -self.w_action_magnitude * float(np.sum(np.square(action)))
        # 动作平滑
        if prev_action is not None and prev_action.size == action.size:
            components["action_smooth"] = -self.w_action_smooth * float(np.sum(np.square(action - prev_action)))
        else:
            components["action_smooth"] = 0.0
        # 跌倒惩罚由 terminated 与 info 中体现，这里可加一项
        if height < self.fall_height_threshold:
            components["fall"] = -10.0
        else:
            components["fall"] = 0.0

        total = sum(components.values())
        return total, components


# =============================================================================
# 4. G1 MPC 权重调节 RL 环境（仿真同步 + 多速率分层）
# =============================================================================

# 多速率分层默认频率：RL(50Hz) : MPC(100Hz) : Sim(1000Hz)
DEFAULT_RL_FREQUENCY_HZ = 50.0
DEFAULT_MPC_FREQUENCY_HZ = 100.0
DEFAULT_SIM_FREQUENCY_HZ = 1000.0


class G1MpcWeightEnv:
    """
    仿真同步、多速率分层的 RL 环境：
    - 不启动 MPC 后台线程；每 RL 步先 set_residual_weights(action)，再在当步内按 MPC 频率执行多次「MPC 求解 + 若干仿真步」。
    - 多速率：RL(50Hz) : MPC(100Hz) : Sim(1000Hz)，即每 RL 步 = 2 次 MPC、每次 MPC 后 10 次仿真步，共 20 次仿真步。
    """

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
        self._rl_frequency_hz = rl_frequency_hz
        self._mpc_frequency_hz = mpc_frequency_hz
        self._sim_frequency_hz = sim_frequency_hz
        self._rl_dt = 1.0 / rl_frequency_hz
        self._mpc_dt = 1.0 / mpc_frequency_hz
        self._sim_dt = 1.0 / sim_frequency_hz
        # 每 RL 步内：MPC 次数、每次 MPC 对应的仿真步数、每 RL 步总仿真步数
        self._mpc_steps_per_rl = int(round(self._rl_dt / self._mpc_dt))
        self._sim_steps_per_mpc = int(round(self._mpc_dt / self._sim_dt))
        self._sim_steps_per_rl = self._mpc_steps_per_rl * self._sim_steps_per_mpc
        assert self._mpc_steps_per_rl >= 1 and self._sim_steps_per_mpc >= 1, "RL/MPC/Sim 频率需满足 rl_dt >= mpc_dt >= sim_dt"

        self._headless = headless
        self._urdf_file = urdf_file
        self._velocity_command = velocity_command if velocity_command is not None else np.array([0.0, 0.0, 0.75, 0.0])
        self._reward_fn = reward_fn or MpcWeightEnvReward()
        self._seed = seed
        if seed is not None:
            np.random.seed(seed)

        # 构建接口与仿真（与 test_mpc_mrt_walk 一致，但不启动 MPC 线程）
        self._interface = mpc_py.WBMpcInterface(task_file, urdf_file, ref_file, setup_ocp=True)
        self._interface.setup_mpc()
        self._state_dim = self._interface.get_state_dim()
        self._weight_module = self._interface.get_weight_adjustment_module()

        # ProceduralMpcMotionManager：速度/高度限幅、滤波、步态
        self._proc_mgr = mpc_py.ProceduralMpcMotionManager(gait_file, ref_file, self._interface)
        init_mpc_state = np.array(self._interface.get_initial_state(), dtype=float)
        initial_base_height = float(init_mpc_state[2])
        cmd = np.array([self._velocity_command[0], self._velocity_command[1], initial_base_height, self._velocity_command[3]])
        self._proc_mgr.set_velocity_command(cmd)

        # 仿真：sim_dt = 1/sim_frequency_hz（1000 Hz）
        model_settings = self._interface.get_model_settings()
        mpc_joint_dim = model_settings.mpc_joint_dim
        robot_description = mpc_py.RobotDescription(urdf_file)
        init_robot_state = mpc_py.RobotState(robot_description, 2)
        base_pos = init_mpc_state[:3]
        base_euler = init_mpc_state[3:6]
        mpc_joint_angles = init_mpc_state[6 : 6 + mpc_joint_dim]
        from scipy.spatial.transform import Rotation as R
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

        # 控制器：不调用 start_mpc_thread，仅用 compute_joint_control_action_sync（MPC 频率用于内部时间步）
        self._controller = mpc_py.WBMpcMrtJointController(self._interface, mpc_frequency_hz)
        self._joint_actions = self._sim.get_robot_joint_action()

        # 空间
        self._obs_dim = get_observation_dim(self._interface)
        self.action_space = MpcResidualActionSpace()
        self.observation_space = type("ObservationSpace", (), {
            "shape": (self._obs_dim,),
            "dtype": np.float32,
        })()

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
        # 若需要从给定状态重置，可由 options 传入；这里做简单重置到初始姿态
        init_mpc_state = self._interface.get_initial_state()
        model_settings = self._interface.get_model_settings()
        mpc_joint_dim = model_settings.mpc_joint_dim
        base_pos = init_mpc_state[:3]
        base_euler = init_mpc_state[3:6]
        mpc_joint_angles = init_mpc_state[6 : 6 + mpc_joint_dim]
        from scipy.spatial.transform import Rotation as R
        quat_xyzw = R.from_euler("zyx", base_euler, degrees=False).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        robot_description = mpc_py.RobotDescription(self._urdf_file)
        init_robot_state = mpc_py.RobotState(robot_description, 2)
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
            np.array([self._velocity_command[0], self._velocity_command[1], float(base_pos[2]), self._velocity_command[3]])
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
        action = self.action_space.clip(action)

        # 1) 注入 RL 残差权重（本 RL 步内所有 MPC 求解前会应用 Q_i = Q_base_i * exp(a_i)）
        self._weight_module.set_residual_weights(action.tolist())

        # 2) 多速率分层：每 RL 步执行 mpc_steps_per_rl 次 MPC，每次 MPC 后执行 sim_steps_per_mpc 次仿真步
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

        # 终止：跌倒
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


# =============================================================================
# 主入口：简短随机策略测试（仿真同步，无 MPC 线程）
# =============================================================================

def main():
    print("=" * 60)
    print("G1 RL 调节 MPC 权重 — 仿真同步多速率环境测试")
    print("=" * 60)

    env = make_g1_mpc_weight_env(headless=(os.environ.get("HEADLESS", "1") == "1"), seed=42)
    obs, info = env.reset(seed=42)
    print(f"观测维度: {obs.shape}, 动作维度: {env.action_space.dim}")

    total_reward = 0.0
    max_steps = 200
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
