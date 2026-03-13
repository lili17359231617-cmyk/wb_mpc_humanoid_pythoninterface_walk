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
        fall_height_threshold: float = 0.6,
        max_roll_pitch: float = 0.5,
        # 新奖励结构的超参数
        sigma_v: float = 0.25,
        sigma_yaw: float = 0.25,
        sigma_stable: float = 0.25,
        w_tracking: float = 1.0,
        w_stable: float = 1.0,
        w_smooth: float = 1.0,
    ):
        self.target_height = target_height
        self.target_vx = target_vx
        self.target_vy = target_vy
        self.target_vyaw = target_vyaw
        self.fall_height_threshold = fall_height_threshold
        self.max_roll_pitch = max_roll_pitch

        self.sigma_v = sigma_v
        self.sigma_yaw = sigma_yaw
        self.sigma_stable = sigma_stable
        self.w_tracking = w_tracking
        self.w_stable = w_stable
        self.w_smooth = w_smooth

    def compute(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        prev_action: Optional[np.ndarray],
        info: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float]]:
        """
        返回 (total_reward, reward_components)。
        obs 布局: base_pos=obs[0:3], base_euler=obs[3:6], joint_pos(6:29), base_vel≈obs[29:32] 等。
        """
        components: Dict[str, float] = {}

        # --- 解析基础量 ---
        height = float(obs[2])
        state_dim = obs.size

        # 线速度与指令
        if state_dim >= 32:
            vx = float(obs[29])
        else:
            vx = 0.0
        vel_cmd = info.get("vel_cmd", None)
        tvx = float(vel_cmd[0]) if vel_cmd is not None else self.target_vx

        # 姿态与偏航角速度
        # obs[3]=yaw, obs[4]=pitch, obs[5]=roll（ZYX 欧拉）
        yaw = float(obs[3])
        pitch = float(obs[4])
        roll = float(obs[5])

        root_ang_vel = info.get("root_ang_vel_local", None)
        if root_ang_vel is not None and len(root_ang_vel) >= 3:
            yaw_rate = float(root_ang_vel[2])
        else:
            yaw_rate = 0.0
        tvyaw = float(vel_cmd[3]) if vel_cmd is not None else self.target_vyaw

        # 是否跌倒
        fallen = bool(info.get("fallen", height < self.fall_height_threshold))

        # --- 1. 追踪奖励 (高斯核) ---
        v_err = vx - tvx
        yaw_err = yaw_rate - tvyaw
        r_tracking = float(
            np.exp(- (v_err ** 2) / self.sigma_v)
            + np.exp(- (yaw_err ** 2) / self.sigma_yaw)
        )
        components["r_tracking"] = r_tracking

        # --- 2. 稳定性奖励 ---
        r_stable = float(
            np.exp(- (roll ** 2 + pitch ** 2) / self.sigma_stable)
        )
        components["r_stable"] = r_stable

        # --- 3. 权重平滑惩罚：基于真实 Q_new 对角线的变化量 ---
        # 优先使用来自 C++ MpcWeightAdjustmentModule 暴露的 Q 对角线；
        # 若不可用，则退化为基于 action 的近似。
        q_diag_now = info.get("Q_diag_now", None)
        q_diag_prev = info.get("Q_diag_prev", None)
        if q_diag_now is not None and q_diag_prev is not None:
            q_now = np.asarray(q_diag_now, dtype=np.float64)
            q_prev = np.asarray(q_diag_prev, dtype=np.float64)
            if q_now.shape == q_prev.shape:
                r_smooth = -0.05 * float(np.sum((q_now - q_prev) ** 2))
            else:
                r_smooth = 0.0
        elif prev_action is not None and prev_action.size == action.size:
            # 退化近似：使用 Q_new ≈ exp(a) 的形式
            q_now = np.exp(action.astype(np.float64))
            q_prev = np.exp(prev_action.astype(np.float64))
            r_smooth = -0.05 * float(np.sum((q_now - q_prev) ** 2))
        else:
            r_smooth = 0.0
        components["r_smooth"] = r_smooth

        # --- 4. 生存奖金 ---
        r_alive = 1.0 if not fallen else 0.0
        components["r_alive"] = r_alive

        # --- 总奖励 ---
        total = (
            self.w_tracking * r_tracking
            + self.w_stable * r_stable
            + self.w_smooth * r_smooth
            + r_alive
        )
        return float(total), components
