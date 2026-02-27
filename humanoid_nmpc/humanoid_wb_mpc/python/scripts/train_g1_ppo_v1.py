import os
import sys

# 1. 将 python/ 根目录加入搜索路径，以便能找到 humanoid_wb_mpc 包
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# 2. 将 C++ 编译产物的路径加入搜索路径
cpp_module_path = "/wb_humanoid_mpc_ws/install/humanoid_wb_mpc/lib/python3.12/site-packages"
if cpp_module_path not in sys.path:
    sys.path.append(cpp_module_path)
# ----------------------------

import gymnasium as gym
from stable_baselines3 import PPO
from humanoid_wb_mpc.envs.G1MpcEnv import G1MpcEnv

def main():
    # 路径定义 (建议使用绝对路径)
    task_p = "/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.info"
    urdf_p = "/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf"
    ref_p  = "/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/robot_models/unitree_g1/g1_wb_mpc/config/command/reference.info"

    # 1. 实例化环境
    print("正在初始化 G1 MPC 强化学习环境...")
    env = G1MpcEnv(task_p, urdf_p, ref_p)

    # 2. 配置算法
    # 对于 70 维的残差控制，建议使用较大的网络层宽
    policy_kwargs = dict(net_arch=[dict(pi=[256, 256], qf=[256, 256])])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./ppo_g1_logs",  # TensorBoard 日志目录
        learning_rate=1e-4,
        n_steps=2048,            # 增加采样步数，让梯度估计更准确
        batch_size=128,          # 增大 batch_size，减少梯度噪声
        n_epochs=10,             # 每一轮采样后学习 10 遍
        gamma=0.99,              # 远期折扣率
        clip_range=0.1,          # 强制限制策略更新幅度
        ent_coef=0.01,           # 增加好奇心，防止过快收敛到“秒跪”
        target_kl=0.01,          # 核心：将 KL 散度强制锁定在 0.01 附近
        policy_kwargs=policy_kwargs,
        device="auto"
    )

    # 3. 开始学习
    print("🚀 启动训练:G1 步态残差优化...")
    model.learn(total_timesteps=1_000_000, progress_bar=True)

    # 4. 保存模型
    model.save("g1_ppo_residual_v1")
    print("✅ 训练模型已保存为 g1_ppo_residual_v1.zip")

if __name__ == "__main__":
    main()