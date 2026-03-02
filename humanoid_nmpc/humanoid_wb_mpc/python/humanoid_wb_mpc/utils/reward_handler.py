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
        返回 (total_reward, reward_components)。
        obs 布局: base_pos=obs[0:3], base_euler=obs[3:6], base_vel≈obs[29:32] 等（与 state_dim 一致）。
        """
        components = {}
        height = float(obs[2])
        components["height"] = -self.w_height * abs(height - self.target_height)

        state_dim = obs.size
        if state_dim >= 32:
            vx, vy = float(obs[29]), float(obs[30])
            components["velocity"] = (
                -self.w_velocity
                * (abs(vx - self.target_vx) + abs(vy - self.target_vy))
            )
        else:
            components["velocity"] = 0.0

        roll, pitch = float(obs[3]), float(obs[4])
        components["orientation"] = -self.w_orientation * (
            min(abs(roll), self.max_roll_pitch) + min(abs(pitch), self.max_roll_pitch)
        )
        components["action_magnitude"] = -self.w_action_magnitude * float(np.sum(np.square(action)))
        if prev_action is not None and prev_action.size == action.size:
            components["action_smooth"] = -self.w_action_smooth * float(np.sum(np.square(action - prev_action)))
        else:
            components["action_smooth"] = 0.0

        if height < self.fall_height_threshold:
            components["fall"] = -10.0
        else:
            components["fall"] = 0.0

        total = sum(components.values())
        return total, components
