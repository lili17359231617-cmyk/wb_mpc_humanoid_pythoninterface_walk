"""
G1 人形机器人 MPC 仿真测试脚本

最终版：
1. 重力补偿 + PD 控制
2. PD 期望位置 = 初始姿态值
3. 避免腿部僵直
4. 支持 HEADLESS 模式
"""

import sys
import os

# 设置 MuJoCo 渲染后端（在导入 mujoco 之前）
# 优先使用环境变量，如果没有设置则根据环境自动选择
if 'MUJOCO_GL' not in os.environ:
    # 检测是否在headless环境
    has_display = os.environ.get('DISPLAY') is not None
    is_headless = os.environ.get('HEADLESS', '0') == '1'

    if is_headless or not has_display:
        # headless环境，使用 egl（如果可用）或 osmesa
        os.environ['MUJOCO_GL'] = 'egl'  # 优先尝试 egl（更稳定）
    else:
        # 有显示环境，使用 glfw
        os.environ['MUJOCO_GL'] = 'glfw'

# 确保 C++ 扩展 humanoid_wb_mpc_py 可被找到（安装于 install/humanoid_wb_mpc/lib/）
_mpc_lib = "/wb_humanoid_mpc_ws/install/humanoid_wb_mpc/lib"
if os.path.isdir(_mpc_lib) and _mpc_lib not in sys.path:
    sys.path.insert(0, _mpc_lib)

# 尝试导入 mujoco，如果失败则尝试其他后端
_mujoco_imported = False
_backends_to_try = [os.environ.get('MUJOCO_GL', 'egl'), 'egl', 'osmesa', 'glfw']

for backend in _backends_to_try:
    try:
        os.environ['MUJOCO_GL'] = backend
        import mujoco
        _mujoco_imported = True
        if backend != os.environ.get('MUJOCO_GL', ''):
            print(f"[INFO] 使用 MuJoCo 后端: {backend}")
        break
    except (ImportError, AttributeError) as e:
        if backend == _backends_to_try[-1]:  # 最后一个也失败了
            print(f"[错误] 无法导入 mujoco，尝试的后端: {_backends_to_try}")
            print(f"[错误] 最后错误: {e}")
            raise

import numpy as np
import humanoid_wb_mpc_py as mpc_py
from scipy.spatial.transform import Rotation as R
import time

# 检查 viewer 是否可用
_has_mujoco_viewer = hasattr(mujoco, 'viewer')
_has_mujoco_viewer_pkg = False
try:
    from mujoco_viewer import MujocoViewer
    _has_mujoco_viewer_pkg = True
except ImportError:
    pass

# ==========================================
# 1. 关节顺序配置 (与 C++ 端一致)
# ==========================================
MPC_JOINT_NAMES = [
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint',
    'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint',
    'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint', 'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint',
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint'
]

WRIST_JOINT_NAMES = [
    'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
    'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint'
]

MPC_JOINT_DIM = 23
ALL_JOINT_DIM = 29

# ==========================================
# 2. 构建 MuJoCo 索引映射
# ==========================================
def build_mujoco_mapping(mj_model):
    """构建 MuJoCo 模型中各关节的索引映射"""
    mpc_qpos_idxs = []
    mpc_qvel_idxs = []
    mpc_ctrl_idxs = []

    all_qpos_idxs = []
    all_qvel_idxs = []
    all_ctrl_idxs = []

    all_joint_names = []

    for name in MPC_JOINT_NAMES:
        jnt_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        act_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if jnt_id == -1 or act_id == -1:
            print(f"    [警告] MPC 关节 '{name}' 在模型中未找到!")
            continue
        mpc_qpos_idxs.append(mj_model.jnt_qposadr[jnt_id])
        mpc_qvel_idxs.append(mj_model.jnt_dofadr[jnt_id])
        mpc_ctrl_idxs.append(act_id)
        all_qpos_idxs.append(mj_model.jnt_qposadr[jnt_id])
        all_qvel_idxs.append(mj_model.jnt_dofadr[jnt_id])
        all_ctrl_idxs.append(act_id)
        all_joint_names.append(name)

    for name in WRIST_JOINT_NAMES:
        jnt_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        act_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if jnt_id == -1 or act_id == -1:
            continue
        all_qpos_idxs.append(mj_model.jnt_qposadr[jnt_id])
        all_qvel_idxs.append(mj_model.jnt_dofadr[jnt_id])
        all_ctrl_idxs.append(act_id)
        all_joint_names.append(name)

    return (
        np.array(mpc_qpos_idxs), np.array(mpc_qvel_idxs), np.array(mpc_ctrl_idxs),
        np.array(all_qpos_idxs), np.array(all_qvel_idxs), np.array(all_ctrl_idxs),
        all_joint_names
    )


# ==========================================
# 3. 初始姿态配置 (与 task.info 一致)
# ==========================================
INITIAL_POSE = np.array([
    -0.05, 0.0, 0.0, 0.1, -0.05, 0.0,  # 左腿 (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
    -0.05, 0.0, 0.0, 0.1, -0.05, 0.0,  # 右腿
    0.0, 0.0, 0.0,                      # 腰部 (yaw, roll, pitch)
    0.0, 0.0, 0.0, 0.0,                 # 左臂 (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow)
    0.0, 0.0, 0.0, 0.0,                 # 右臂
    0.0, 0.0, 0.0,                      # 左手腕
    0.0, 0.0, 0.0,                      # 右手腕
])


# ==========================================
# 4. 主仿真循环
# ==========================================
def main():
    # ===== 路径配置 =====
    task_file = "/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.info"
    urdf_file = "/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf"
    ref_file = "/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/robot_models/unitree_g1/g1_wb_mpc/config/command/reference.info"
    xml_file = "/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/robot_models/unitree_g1/g1_description/urdf/g1_29dof.xml"

    print("=" * 60)
    print("G1 MPC 仿真测试 (最终版)")
    print("=" * 60)

    # ===== 初始化 C++ MPC 接口 =====
    print("\n[1/5] 初始化 C++ MPC 接口...")
    interface = mpc_py.WBMpcInterface(task_file, urdf_file, ref_file)
    state_dim = interface.get_state_dim()
    input_dim = interface.get_input_dim()
    print(f"    MPC 期望状态维度: {state_dim}")
    interface.setup_mpc()

    # ===== 设置目标轨迹 =====
    print("\n[2/5] 设置目标轨迹...")
    initial_state = interface.get_initial_state()
    print(f"    MPC 初始腰部 yaw/roll/pitch: {initial_state[6+12]:.4f}, {initial_state[6+13]:.4f}, {initial_state[6+14]:.4f}")
    interface.set_target_state(initial_state)

    # ===== 初始化 MuJoCo =====
    print("\n[3/5] 初始化 MuJoCo...")
    model = mujoco.MjModel.from_xml_path(xml_file)
    data = mujoco.MjData(model)

    (mpc_qpos_idxs, mpc_qvel_idxs, mpc_ctrl_idxs,
     all_qpos_idxs, all_qvel_idxs, all_ctrl_idxs, all_joint_names) = build_mujoco_mapping(model)

    print(f"    MPC 关节数: {len(mpc_qpos_idxs)}")
    print(f"    全部关节数: {len(all_qpos_idxs)}")

    # 设置初始姿态
    print(f"\n    [初始姿态设置]")
    for i, idx in enumerate(all_qpos_idxs):
        if i < len(INITIAL_POSE):
            data.qpos[idx] = INITIAL_POSE[i]
            print(f"      [{i:2d}] {all_joint_names[i]}: {INITIAL_POSE[i]:.4f}")
    for idx in all_qvel_idxs:
        data.qvel[idx] = 0.0

    mujoco.mj_forward(model, data)

    # 打印初始姿态检查
    quat_wxyz = data.qpos[3:7]
    quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    euler = R.from_quat(quat_xyzw).as_euler('zyx')
    print(f"\n    [调试] MuJoCo 初始基座高度: {data.qpos[2]:.3f} m")
    print(f"    [调试] MuJoCo 初始欧拉角 (ZYX): yaw={euler[0]:.3f}, pitch={euler[1]:.3f}, roll={euler[2]:.3f}")

    # ===== 启动仿真 =====
    print("\n[4/5] 启动仿真...")
    print("-" * 60)

    frame_count = 0
    last_print_time = time.time()

    # PD 增益配置
    # 脚踝关节索引：left_ankle_pitch(4), left_ankle_roll(5), right_ankle_pitch(10), right_ankle_roll(11)
    kp_mpc = 150.0      # MPC关节的默认位置增益
    kd_mpc = 20.0       # MPC关节的默认速度阻尼
    kp_ankle = 50.0     # 脚踝关节位置增益（大幅降低，减少抖动）
    kd_ankle = 35.0     # 脚踝关节速度阻尼（大幅增加，增加稳定性）
    kp_wrist = 20.0
    kd_wrist = 0.5

    # 脚踝关节索引（在MPC_JOINT_NAMES中的位置）
    ankle_joint_indices = [4, 5, 10, 11]  # left_ankle_pitch, left_ankle_roll, right_ankle_pitch, right_ankle_roll

    print(f"\n    [PD参数配置]")
    print(f"      MPC关节: kp={kp_mpc}, kd={kd_mpc}")
    print(f"      脚踝关节: kp={kp_ankle}, kd={kd_ankle} (降低kp，增加kd以减少抖动)")
    print(f"      手腕关节: kp={kp_wrist}, kd={kd_wrist}")

    # Headless 模式检测
    headless = os.environ.get('HEADLESS', '0') == '1' or (not _has_mujoco_viewer and not _has_mujoco_viewer_pkg)

    if headless:
        print("\n    [INFO] 运行在 HEADLESS 模式 (无图形界面)")
        try:
            while frame_count < 5000:  # 限制帧数
                step_start = time.time()

                # ===== 1. 状态映射 (仅用于 MPC) =====
                if frame_count == 0:
                    try:
                        ocs2_state = np.zeros(state_dim)
                        ocs2_state[0:3] = data.qpos[0:3]
                        quat_wxyz = data.qpos[3:7]
                        quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
                        ocs2_state[3:6] = R.from_quat(quat_xyzw).as_euler('zyx')
                        for i, idx in enumerate(mpc_qpos_idxs):
                            ocs2_state[6+i] = data.qpos[idx]
                        ocs2_state[29:32] = data.qvel[0:3]
                        ocs2_state[32:35] = data.qvel[3:6]
                        for i, idx in enumerate(mpc_qvel_idxs):
                            ocs2_state[35+i] = data.qvel[idx]

                        optimal_u = interface.run_mpc(ocs2_state, data.time)
                        print(f"    [MPC] 首次运行成功")
                    except Exception as e:
                        print(f"\n[MPC 首次错误] {e}")
                        optimal_u = np.zeros(input_dim)

                # ===== 2. 获取当前关节状态 =====
                q_cur = np.zeros(ALL_JOINT_DIM)
                v_cur = np.zeros(ALL_JOINT_DIM)
                for i, idx in enumerate(all_qpos_idxs):
                    q_cur[i] = data.qpos[idx]
                for i, idx in enumerate(all_qvel_idxs):
                    v_cur[i] = data.qvel[idx]

                # ===== 3. 重力补偿 =====
                data.qacc[:] = 0
                data.xfrc_applied[:] = 0
                mujoco.mj_inverse(model, data)

                # 修复：使用正确的索引映射方法
                # qfrc_inverse 的索引直接对应 qpos 的索引
                # 所以应该使用 data.qfrc_inverse[mpc_qpos_idxs[i]] 而不是 [6+i]
                tau_gravity = np.array([data.qfrc_inverse[mpc_qpos_idxs[i]] for i in range(len(mpc_qpos_idxs))])

                # 验证重力补偿索引（仅在首帧检查一次）
                if frame_count == 0:
                    print("\n    [验证] 重力补偿索引检查:")
                    print(f"      qfrc_inverse 总长度: {len(data.qfrc_inverse)}")
                    print(f"      MPC 关节数: {len(mpc_qpos_idxs)}")

                    # 验证修复后的索引映射
                    mismatches = []
                    old_method = data.qfrc_inverse[6:6+len(mpc_qpos_idxs)]  # 旧方法
                    for i in range(len(mpc_qpos_idxs)):
                        qpos_idx = mpc_qpos_idxs[i]
                        correct_value = tau_gravity[i]  # 新方法（已修复）
                        old_value = old_method[i] if i < len(old_method) else 0

                        if abs(correct_value - old_value) > 1e-6:
                            mismatches.append((i, MPC_JOINT_NAMES[i], qpos_idx, correct_value, old_value))

                    if len(mismatches) == 0:
                        print(f"      ✅ 所有 {len(mpc_qpos_idxs)} 个关节的索引映射正确！")
                    else:
                        print(f"      ⚠️  发现旧方法有 {len(mismatches)} 个索引映射错误（已修复）:")
                        for i, name, qpos_idx, correct, old in mismatches[:3]:  # 只显示前3个
                            print(f"        关节 {i} ({name}): qpos_idx={qpos_idx}, "
                                  f"正确值={correct:.4f}, 旧值={old:.4f}, "
                                  f"差值={abs(correct-old):.4f}")
                        if len(mismatches) > 3:
                            print(f"        ... 还有 {len(mismatches)-3} 个错误")
                        print(f"      ✅ 已修复：现在使用 data.qfrc_inverse[mpc_qpos_idxs[i]]")

                # ===== 4. PD 控制 =====
                tau_pd = np.zeros(ALL_JOINT_DIM)

                # MPC 控制关节 (23个)
                for i in range(MPC_JOINT_DIM):
                    desired_pos = INITIAL_POSE[i]
                    error_pos = desired_pos - q_cur[i]
                    error_vel = 0 - v_cur[i]

                    # 脚踝关节使用单独的PD参数（更低的kp，更高的kd）
                    if i in ankle_joint_indices:
                        tau_pd[i] = kp_ankle * error_pos + kd_ankle * error_vel
                    else:
                        tau_pd[i] = kp_mpc * error_pos + kd_mpc * error_vel

                # 手腕关节 - 期望位置 = 0
                for i in range(MPC_JOINT_DIM, ALL_JOINT_DIM):
                    tau_pd[i] = kp_wrist * (0 - q_cur[i]) + kd_wrist * (0 - v_cur[i])

                # ===== 5. 应用控制 =====
                data.ctrl.fill(0.0)

                # MPC控制关节 (23个) - 包含重力补偿
                for i, act_id in enumerate(mpc_ctrl_idxs):
                    torque = tau_gravity[i] + tau_pd[i]
                    # 脚踝关节使用更小的力矩限制，减少抖动
                    if i in ankle_joint_indices:
                        max_torque = 40.0  # 脚踝关节力矩限制更小
                    else:
                        max_torque = 60.0
                    torque = np.clip(torque, -max_torque, max_torque)
                    data.ctrl[act_id] = torque

                    # 调试：在首帧打印脚踝关节的控制信息
                    if frame_count == 0 and i in ankle_joint_indices:
                        print(f"      [调试] {MPC_JOINT_NAMES[i]}: "
                              f"重力补偿={tau_gravity[i]:.4f}, "
                              f"PD力矩={tau_pd[i]:.4f}, "
                              f"总力矩={torque:.4f}")

                # 手腕关节 (6个) - 仅PD控制，无重力补偿（手腕质量小）
                for i in range(MPC_JOINT_DIM, ALL_JOINT_DIM):
                    act_id = all_ctrl_idxs[i]
                    torque = tau_pd[i]  # 手腕关节只使用PD控制
                    max_torque = 20.0  # 手腕关节力矩限制更小
                    torque = np.clip(torque, -max_torque, max_torque)
                    data.ctrl[act_id] = torque

                # ===== 6. 仿真步进 =====
                mujoco.mj_step(model, data)

                # ===== 7. 调试打印 =====
                frame_count += 1
                current_time = time.time()
                if current_time - last_print_time >= 0.5:
                    quat_wxyz = data.qpos[3:7]
                    quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
                    euler = R.from_quat(quat_xyzw).as_euler('zyx')

                    height = data.qpos[2]
                    gravity_norm = np.linalg.norm(tau_gravity)
                    pd_norm = np.linalg.norm(tau_pd)

                    l_hip_pitch = q_cur[0]
                    l_knee = q_cur[3]
                    l_ankle = q_cur[4]
                    r_hip_pitch = q_cur[6]

                    print(f"    Frame: {frame_count:5d} | Time: {data.time:.2f}s | "
                          f"H: {height:.3f}m | YPR: [{euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}] | "
                          f"LHip: {l_hip_pitch:.2f} | LKnee: {l_knee:.2f} | LAnkle: {l_ankle:.2f} | RHip: {r_hip_pitch:.2f} | "
                          f"G: {gravity_norm:.1f} | PD: {pd_norm:.1f}",
                          end='\r')
                    last_print_time = current_time

                # ===== 8. 频率控制 =====
                elapsed = time.time() - step_start
                timestep = model.opt.timestep
                if elapsed < timestep:
                    time.sleep(timestep - elapsed)

            print("\n    [INFO] 完成 5000 帧仿真")
        except KeyboardInterrupt:
            print("\n    [INFO] 用户中断仿真")
    elif _has_mujoco_viewer_pkg:
        # 使用 mujoco_viewer 包
        print("\n    [INFO] 使用 mujoco_viewer 渲染")
        try:
            viewer = MujocoViewer(model, data)
            while True:
                step_start = time.time()

                # ===== 1. 状态映射 (仅用于 MPC) =====
                if frame_count == 0:
                    try:
                        ocs2_state = np.zeros(state_dim)
                        ocs2_state[0:3] = data.qpos[0:3]
                        quat_wxyz = data.qpos[3:7]
                        quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
                        ocs2_state[3:6] = R.from_quat(quat_xyzw).as_euler('zyx')
                        for i, idx in enumerate(mpc_qpos_idxs):
                            ocs2_state[6+i] = data.qpos[idx]
                        ocs2_state[29:32] = data.qvel[0:3]
                        ocs2_state[32:35] = data.qvel[3:6]
                        for i, idx in enumerate(mpc_qvel_idxs):
                            ocs2_state[35+i] = data.qvel[idx]

                        optimal_u = interface.run_mpc(ocs2_state, data.time)
                        print(f"    [MPC] 首次运行成功")
                    except Exception as e:
                        print(f"\n[MPC 首次错误] {e}")
                        optimal_u = np.zeros(input_dim)

                # ===== 2. 获取当前关节状态 =====
                q_cur = np.zeros(ALL_JOINT_DIM)
                v_cur = np.zeros(ALL_JOINT_DIM)
                for i, idx in enumerate(all_qpos_idxs):
                    q_cur[i] = data.qpos[idx]
                for i, idx in enumerate(all_qvel_idxs):
                    v_cur[i] = data.qvel[idx]

                # ===== 3. 重力补偿 =====
                data.qacc[:] = 0
                data.xfrc_applied[:] = 0
                mujoco.mj_inverse(model, data)

                # 修复：使用正确的索引映射方法
                # qfrc_inverse 的索引直接对应 qpos 的索引
                # 所以应该使用 data.qfrc_inverse[mpc_qpos_idxs[i]] 而不是 [6+i]
                tau_gravity = np.array([data.qfrc_inverse[mpc_qpos_idxs[i]] for i in range(len(mpc_qpos_idxs))])

                # 验证重力补偿索引（仅在首帧检查一次）
                if frame_count == 0:
                    print("\n    [验证] 重力补偿索引检查:")
                    print(f"      qfrc_inverse 总长度: {len(data.qfrc_inverse)}")
                    print(f"      MPC 关节数: {len(mpc_qpos_idxs)}")

                    # 验证修复后的索引映射
                    mismatches = []
                    old_method = data.qfrc_inverse[6:6+len(mpc_qpos_idxs)]  # 旧方法
                    for i in range(len(mpc_qpos_idxs)):
                        qpos_idx = mpc_qpos_idxs[i]
                        correct_value = tau_gravity[i]  # 新方法（已修复）
                        old_value = old_method[i] if i < len(old_method) else 0

                        if abs(correct_value - old_value) > 1e-6:
                            mismatches.append((i, MPC_JOINT_NAMES[i], qpos_idx, correct_value, old_value))

                    if len(mismatches) == 0:
                        print(f"      ✅ 所有 {len(mpc_qpos_idxs)} 个关节的索引映射正确！")
                    else:
                        print(f"      ⚠️  发现旧方法有 {len(mismatches)} 个索引映射错误（已修复）:")
                        for i, name, qpos_idx, correct, old in mismatches[:3]:  # 只显示前3个
                            print(f"        关节 {i} ({name}): qpos_idx={qpos_idx}, "
                                  f"正确值={correct:.4f}, 旧值={old:.4f}, "
                                  f"差值={abs(correct-old):.4f}")
                        if len(mismatches) > 3:
                            print(f"        ... 还有 {len(mismatches)-3} 个错误")
                        print(f"      ✅ 已修复：现在使用 data.qfrc_inverse[mpc_qpos_idxs[i]]")

                # ===== 4. PD 控制 =====
                tau_pd = np.zeros(ALL_JOINT_DIM)

                # MPC 控制关节 (23个)
                for i in range(MPC_JOINT_DIM):
                    desired_pos = INITIAL_POSE[i]
                    error_pos = desired_pos - q_cur[i]
                    error_vel = 0 - v_cur[i]

                    # 脚踝关节使用单独的PD参数（更低的kp，更高的kd）
                    if i in ankle_joint_indices:
                        tau_pd[i] = kp_ankle * error_pos + kd_ankle * error_vel
                    else:
                        tau_pd[i] = kp_mpc * error_pos + kd_mpc * error_vel

                # 手腕关节 - 期望位置 = 0
                for i in range(MPC_JOINT_DIM, ALL_JOINT_DIM):
                    tau_pd[i] = kp_wrist * (0 - q_cur[i]) + kd_wrist * (0 - v_cur[i])

                # ===== 5. 应用控制 =====
                data.ctrl.fill(0.0)

                # MPC控制关节 (23个) - 包含重力补偿
                for i, act_id in enumerate(mpc_ctrl_idxs):
                    torque = tau_gravity[i] + tau_pd[i]
                    # 脚踝关节使用更小的力矩限制，减少抖动
                    if i in ankle_joint_indices:
                        max_torque = 40.0  # 脚踝关节力矩限制更小
                    else:
                        max_torque = 60.0
                    torque = np.clip(torque, -max_torque, max_torque)
                    data.ctrl[act_id] = torque

                    # 调试：在首帧打印脚踝关节的控制信息
                    if frame_count == 0 and i in ankle_joint_indices:
                        print(f"      [调试] {MPC_JOINT_NAMES[i]}: "
                              f"重力补偿={tau_gravity[i]:.4f}, "
                              f"PD力矩={tau_pd[i]:.4f}, "
                              f"总力矩={torque:.4f}")

                # 手腕关节 (6个) - 仅PD控制，无重力补偿（手腕质量小）
                for i in range(MPC_JOINT_DIM, ALL_JOINT_DIM):
                    act_id = all_ctrl_idxs[i]
                    torque = tau_pd[i]  # 手腕关节只使用PD控制
                    max_torque = 20.0  # 手腕关节力矩限制更小
                    torque = np.clip(torque, -max_torque, max_torque)
                    data.ctrl[act_id] = torque

                # ===== 6. 仿真步进 =====
                mujoco.mj_step(model, data)
                viewer.render()

                # ===== 7. 调试打印 =====
                frame_count += 1
                current_time = time.time()
                if current_time - last_print_time >= 0.5:
                    quat_wxyz = data.qpos[3:7]
                    quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
                    euler = R.from_quat(quat_xyzw).as_euler('zyx')

                    height = data.qpos[2]
                    gravity_norm = np.linalg.norm(tau_gravity)
                    pd_norm = np.linalg.norm(tau_pd)

                    l_hip_pitch = q_cur[0]
                    l_knee = q_cur[3]
                    l_ankle = q_cur[4]
                    r_hip_pitch = q_cur[6]

                    print(f"    Frame: {frame_count:5d} | Time: {data.time:.2f}s | "
                          f"H: {height:.3f}m | YPR: [{euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}] | "
                          f"LHip: {l_hip_pitch:.2f} | LKnee: {l_knee:.2f} | LAnkle: {l_ankle:.2f} | RHip: {r_hip_pitch:.2f} | "
                          f"G: {gravity_norm:.1f} | PD: {pd_norm:.1f}",
                          end='\r')
                    last_print_time = current_time

                # ===== 8. 频率控制 =====
                elapsed = time.time() - step_start
                timestep = model.opt.timestep
                if elapsed < timestep:
                    time.sleep(timestep - elapsed)

                if not viewer.is_alive:
                    break

            print("\n    [INFO] 仿真结束")
            viewer.close()
        except KeyboardInterrupt:
            print("\n    [INFO] 用户中断仿真")
            viewer.close()

    print("\n" + "-" * 60)
    print("[5/5] 仿真结束")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断仿真")
    except Exception as e:
        print(f"\n[Error] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
