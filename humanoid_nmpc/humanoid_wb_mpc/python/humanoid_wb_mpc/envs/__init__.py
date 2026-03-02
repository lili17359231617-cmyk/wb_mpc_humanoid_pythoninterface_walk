from .G1MpcWeightEnv import (
    G1MpcWeightEnv,
    make_g1_mpc_weight_env,
    robot_state_to_observation,
    get_observation_dim,
)

# G1MpcEnv 依赖 Python mujoco，延迟导入以免无头环境下触发 OpenGL 错误
_g1_mpc_registered = False


def __getattr__(name):
    global _g1_mpc_registered
    if name == "G1MpcEnv":
        from .G1MpcEnv import G1MpcEnv
        if not _g1_mpc_registered:
            _register_g1_mpc()
            _g1_mpc_registered = True
        return G1MpcEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _register_g1_mpc():
    from gymnasium.envs.registration import register
    try:
        register(
            id='G1Mpc-v0',
            entry_point='humanoid_wb_mpc.envs.G1MpcEnv:G1MpcEnv',
            kwargs={
                'task_file': '/path/to/task.info',
                'urdf_file': '/path/to/g1_29dof.urdf',
                'ref_file': '/path/to/reference.info'
            }
        )
    except Exception:
        pass  # 可能已注册


__all__ = [
    "G1MpcEnv",
    "G1MpcWeightEnv",
    "make_g1_mpc_weight_env",
    "robot_state_to_observation",
    "get_observation_dim",
]