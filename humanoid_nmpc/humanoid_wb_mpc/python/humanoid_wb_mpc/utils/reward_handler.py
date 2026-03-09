"""
MPC 权重 RL 环境奖励函数

- MpcWeightEnvReward: 跟踪高度/速度、姿态稳定、动作幅值/平滑惩罚、跌倒惩罚。
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any


class MpcWeightEnvReward:
    """
    奖励函数：鼓励跟踪指令、保持高度与姿态，惩罚跌倒与过大/剧烈变化的权重。
    观测与 MPC 状态同构: [base_pos(3), base_euler(3), joint_pos(23), base_vel(3), euler_dot(3), joint_vel(23)]
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
        w_survival: float = 0.1,
        fall_height_threshold: float = 0.6,
        fall_penalty: float = -2.0,
        max_roll_pitch: float = 0.5,
        w_gait_symmetry: float = 0.0,
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
        self.w_survival = w_survival
        self.fall_height_threshold = fall_height_threshold
        self.fall_penalty = fall_penalty
        self.max_roll_pitch = max_roll_pitch
        self.w_gait_symmetry = w_gait_symmetry

    def compute(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        prev_action: Optional[np.ndarray],
        info: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float]]:
        """
        返回 (total_reward, reward_components)。
        obs 布局: base_pos=obs[0:3], base_euler=obs[3:6], base_vel≈obs[29:32] 等（与 state_dim 一致）。
        """
        components = {}
        height = float(obs[2])
        components["height"] = -self.w_height * abs(height - self.target_height)

        state_dim = obs.size
        if state_dim >= 32:
            vx, vy = float(obs[29]), float(obs[30])
            vel_cmd = info.get("vel_cmd", None)
            tvx = float(vel_cmd[0]) if vel_cmd is not None else self.target_vx
            tvy = float(vel_cmd[1]) if vel_cmd is not None else self.target_vy
            components["velocity"] = (
                -self.w_velocity
                * (abs(vx - tvx) + abs(vy - tvy))
            )
        else:
            components["velocity"] = 0.0

        # obs[3]=yaw, obs[4]=pitch, obs[5]=roll（ZYX欧拉角顺序，与 task.info initialState 一致）
        pitch, roll = float(obs[4]), float(obs[5])
        components["orientation"] = -self.w_orientation * (
            min(abs(roll), self.max_roll_pitch) + min(abs(pitch), self.max_roll_pitch)
        )
        components["survival"] = self.w_survival
        components["action_magnitude"] = -self.w_action_magnitude * float(np.sum(np.square(action)))
        if prev_action is not None and prev_action.size == action.size:
            components["action_smooth"] = -self.w_action_smooth * float(np.sum(np.square(action - prev_action)))
        else:
            components["action_smooth"] = 0.0

        if height < self.fall_height_threshold:
            components["fall"] = self.fall_penalty
        else:
            components["fall"] = 0.0

        # 步态对称性奖励（Phase 2 启用）
        # 关节顺序（obs[6:12]=左腿, obs[12:18]=右腿）：
        #   hip_pitch(同号), hip_roll(异号), hip_yaw(异号), knee(同号), ankle_pitch(同号), ankle_roll(异号)
        if self.w_gait_symmetry > 0.0 and obs.size >= 18:
            sym_signs = np.array([1.0, -1.0, -1.0, 1.0, 1.0, -1.0], dtype=np.float64)
            left_leg = obs[6:12].astype(np.float64)
            right_leg = obs[12:18].astype(np.float64)
            symmetry_err = float(np.sum(np.abs(left_leg - sym_signs * right_leg)))
            components["gait_symmetry"] = -self.w_gait_symmetry * symmetry_err
        else:
            components["gait_symmetry"] = 0.0

        total = sum(components.values())
        return total, components
