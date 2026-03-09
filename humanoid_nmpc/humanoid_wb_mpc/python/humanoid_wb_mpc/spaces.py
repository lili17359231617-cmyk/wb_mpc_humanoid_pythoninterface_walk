"""
RL 动作/观测空间等核心逻辑（MPC 权重调节用）

- MpcResidualActionSpace: 58 维残差权重动作空间，与 MpcWeightAdjustmentModule 一致。
"""

import numpy as np
from humanoid_wb_mpc.config import RESIDUAL_WEIGHT_DIM


class MpcResidualActionSpace:
    """
    RL 动作空间：与 MPC 状态/Q 维一致的残差权重维度。
    在 C++ 侧映射为 Q_i = Q_base_i * exp(a_i)，故 a 通常取较小范围（如 [-1, 1]）即可。
    """

    def __init__(self, dim: int = None, low: float = -0.4, high: float = 0.4):
        self.dim = dim if dim is not None else RESIDUAL_WEIGHT_DIM
        self.low = np.full(self.dim, low, dtype=np.float32)
        self.high = np.full(self.dim, high, dtype=np.float32)

    def sample(self) -> np.ndarray:
        return np.random.uniform(self.low, self.high).astype(np.float32)

    def clip(self, action: np.ndarray) -> np.ndarray:
        return np.clip(action, self.low, self.high)
