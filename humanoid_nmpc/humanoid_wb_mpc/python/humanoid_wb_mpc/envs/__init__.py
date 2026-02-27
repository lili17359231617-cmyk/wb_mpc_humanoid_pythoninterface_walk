from gymnasium.envs.registration import register
from .G1MpcEnv import G1MpcEnv

register(
    id='G1Mpc-v0',
    entry_point='humanoid_wb_mpc.envs.G1MpcEnv:G1MpcEnv',
    kwargs={
        'task_file': '/path/to/task.info',
        'urdf_file': '/path/to/g1_29dof.urdf',
        'ref_file': '/path/to/reference.info'
    }
)