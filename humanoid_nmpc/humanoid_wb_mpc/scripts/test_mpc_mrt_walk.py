"""
G1 人形机器人 MPC-MRT 控制测试脚本 - 稳定行走版本

实现完整的数据流：
1. 初始化：通过 WBMpcInterface 加载模型，通过 MujocoSimInterface 启动仿真
2. 感知：从仿真器获取 RobotState
3. 决策：WBMpcInterface 在后台线程计算最优路径；通过键盘输入动态修改速度指令
4. 执行：WBMpcMrtJointController 将 MPC 结果转换为 RobotJointAction
5. 驱动：将动作作用于 MuJoCo 里的机器人，循环往复

新增功能：
- 使用虚拟显示器（Xvfb）+ 键盘输入库动态修改速度指令（行走方向、转动角度、站立高度）
- 结合当前机器人状态，将速度指令积分生成期望基座位置和速度
- 摆动腿规划器根据当前相位规划脚在空中的期望轨迹
- 将新轨迹写入 MPC 的线程安全缓冲区
"""

import sys
import os
import time
import numpy as np
import threading
from scipy.spatial.transform import Rotation as R

# 设置 MuJoCo 渲染后端
if 'MUJOCO_GL' not in os.environ:
    has_display = os.environ.get('DISPLAY') is not None
    is_headless = os.environ.get('HEADLESS', '0') == '1'
    if is_headless or not has_display:
        os.environ['MUJOCO_GL'] = 'egl'
    else:
        os.environ['MUJOCO_GL'] = 'glfw'

# 确保 C++ 扩展可被找到
_mpc_lib = "/wb_humanoid_mpc_ws/install/humanoid_wb_mpc/lib"
if os.path.isdir(_mpc_lib) and _mpc_lib not in sys.path:
    sys.path.insert(0, _mpc_lib)

try:
    import humanoid_wb_mpc_py as mpc_py
except ImportError as e:
    print(f"[错误] 无法导入 humanoid_wb_mpc_py: {e}")
    print(f"请确保已编译并安装 Python 绑定模块")
    sys.exit(1)

# 尝试导入键盘输入库
try:
    from pynput import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    try:
        import keyboard as kb_module
        KEYBOARD_AVAILABLE = True
        keyboard = kb_module
    except ImportError:
        KEYBOARD_AVAILABLE = False
        print("[警告] 未安装键盘输入库 (pynput 或 keyboard)，将使用默认速度指令")


class VelocityCommand:
    """速度指令管理器（线程安全）"""
    def __init__(self, control_file=None):
        self.lock = threading.Lock()
        # 速度指令: [v_x, v_y, v_z, v_yaw]
        # 注意: v_z 是目标高度（m），不是速度！
        # 初始值使用 reference.info 中的 defaultBaseHeight = 0.7925m
        # 或 task.info 中的初始高度 p_base_z = 0.8m
        self.commanded_velocities = np.array([0.0, 0.0, 0.8, 0.0])  # 初始高度设为0.8m（与task.info一致）
        # 速度增量
        self.v_x_step = 0.1  # m/s
        self.v_y_step = 0.1  # m/s
        self.v_z_step = 0.02  # m (高度变化)
        self.v_yaw_step = 0.2  # rad/s

        # 文件控制模式（用于Docker容器）
        self.control_file = control_file
        if control_file:
            # 创建控制文件，初始高度使用0.8m（与task.info的p_base_z一致）
            os.makedirs(os.path.dirname(control_file) if os.path.dirname(control_file) else '.', exist_ok=True)
            with open(control_file, 'w') as f:
                f.write("0.0,0.0,0.8,0.0\n")  # v_x, v_y, v_z, v_yaw（初始高度0.8m）
            print(f"[文件控制] 使用控制文件: {control_file}")
            print(f"  格式: v_x,v_y,v_z,v_yaw (每行一个命令)")
            print(f"  示例: 0.2,0.0,0.8,0.0  (前进，高度0.8m)")

    def get_velocities(self):
        """获取当前速度指令（线程安全）"""
        # 如果使用文件控制，从文件读取
        if self.control_file and os.path.exists(self.control_file):
            try:
                with open(self.control_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        # 读取最后一行
                        last_line = lines[-1].strip()
                        if last_line:
                            parts = last_line.split(',')
                            if len(parts) == 4:
                                try:
                                    v_x = float(parts[0])
                                    v_y = float(parts[1])
                                    v_z = float(parts[2])
                                    v_yaw = float(parts[3])
                                    with self.lock:
                                        old_vel = self.commanded_velocities.copy()
                                        self.commanded_velocities = np.array([
                                            np.clip(v_x, -0.5, 0.5),
                                            np.clip(v_y, -0.3, 0.3),
                                            np.clip(v_z, 0.7, 1.0),
                                            np.clip(v_yaw, -0.5, 0.5)
                                        ])
                                        # 调试：如果速度变化，打印信息
                                        if not np.allclose(old_vel, self.commanded_velocities, atol=0.001):
                                            print(f"[文件控制] 读取到新速度指令: "
                                                  f"v_x={self.commanded_velocities[0]:.2f}, "
                                                  f"v_y={self.commanded_velocities[1]:.2f}, "
                                                  f"v_z={self.commanded_velocities[2]:.2f}m, "
                                                  f"v_yaw={self.commanded_velocities[3]:.2f}")
                                except ValueError as e:
                                    print(f"[文件控制] 警告: 无法解析速度指令 '{last_line}': {e}")
            except Exception as e:
                print(f"[文件控制] 警告: 读取控制文件失败: {e}")

        with self.lock:
            return self.commanded_velocities.copy()

    def update_velocity(self, delta_v_x=0.0, delta_v_y=0.0, delta_v_z=0.0, delta_v_yaw=0.0):
        """更新速度指令（线程安全）"""
        with self.lock:
            self.commanded_velocities[0] += delta_v_x
            self.commanded_velocities[1] += delta_v_y
            self.commanded_velocities[2] += delta_v_z
            self.commanded_velocities[3] += delta_v_yaw
            # 限制速度范围
            self.commanded_velocities[0] = np.clip(self.commanded_velocities[0], -0.5, 0.5)  # v_x: -0.5 ~ 0.5 m/s
            self.commanded_velocities[1] = np.clip(self.commanded_velocities[1], -0.3, 0.3)  # v_y: -0.3 ~ 0.3 m/s
            self.commanded_velocities[2] = np.clip(self.commanded_velocities[2], 0.7, 1.0)  # v_z: 0.7 ~ 1.0 m (高度)
            self.commanded_velocities[3] = np.clip(self.commanded_velocities[3], -0.5, 0.5)  # v_yaw: -0.5 ~ 0.5 rad/s


def get_euler_angles_zyx_derivatives_from_local_angular_velocity(euler_angles, angular_velocity_local):
    """
    从本地角速度计算ZYX欧拉角导数

    参数:
        euler_angles: ZYX欧拉角 [z, y, x]
        angular_velocity_local: 本地坐标系中的角速度 [wx, wy, wz]

    返回:
        欧拉角导数 [dz, dy, dx]
    """
    sy = np.sin(euler_angles[1])
    cy = np.cos(euler_angles[1])
    sx = np.sin(euler_angles[2])
    cx = np.cos(euler_angles[2])
    wx = angular_velocity_local[0]
    wy = angular_velocity_local[1]
    wz = angular_velocity_local[2]

    # 根据 getEulerAnglesZyxDerivativesFromLocalAngularVelocity 实现
    tmp = sx * wy / cy + cx * wz / cy
    return np.array([tmp, cx * wy - sx * wz, wx + sy * tmp])


def robot_state_to_mpc_state(robot_state, interface, robot_description):
    """
    将 RobotState 转换为 MPC 状态向量

    参数:
        robot_state: RobotState 对象
        interface: WBMpcInterface 对象
        robot_description: RobotDescription 对象

    返回:
        MPC 状态向量 (numpy array)
    """
    model_settings = interface.get_model_settings()
    mpc_joint_dim = model_settings.mpc_joint_dim
    mpc_joint_names = model_settings.mpc_model_joint_names

    # MPC 状态向量结构: [base_pos(3), base_euler(3), joint_angles(n), base_vel_world(3), euler_derivatives(3), joint_vels(n)]
    state_dim = interface.get_state_dim()
    mpc_state = np.zeros(state_dim)

    # 基座位置 [0:3] (世界坐标系)
    base_pos = robot_state.get_root_position()
    mpc_state[0:3] = base_pos

    # 基座姿态：从四元数转换为欧拉角 [3:6] (ZYX顺序)
    quat_wxyz = robot_state.get_root_rotation_quat()  # [w, x, y, z]
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])  # 转换为 [x, y, z, w]
    euler = R.from_quat(quat_xyzw).as_euler('zyx')  # ZYX顺序: [z, y, x]
    mpc_state[3:6] = euler

    # 关节角度 [6:6+mpc_joint_dim]
    for i, joint_name in enumerate(mpc_joint_names):
        try:
            joint_idx = robot_description.get_joint_index(joint_name)
            mpc_state[6 + i] = robot_state.get_joint_position(joint_idx)
        except Exception as e:
            print(f"    ⚠ 无法获取关节 {joint_name} 的位置: {e}")

    # 基座线速度（从本地坐标系转换到世界坐标系）[6+mpc_joint_dim:9+mpc_joint_dim]
    base_linear_vel_local = robot_state.get_root_linear_velocity()
    # 将本地速度转换到世界坐标系
    quat_rot = R.from_quat(quat_xyzw)
    base_linear_vel_world = quat_rot.apply(base_linear_vel_local)
    mpc_state[6+mpc_joint_dim:9+mpc_joint_dim] = base_linear_vel_world

    # 欧拉角导数（从本地角速度计算）[9+mpc_joint_dim:12+mpc_joint_dim]
    base_angular_vel_local = robot_state.get_root_angular_velocity()
    euler_derivatives = get_euler_angles_zyx_derivatives_from_local_angular_velocity(euler, base_angular_vel_local)
    mpc_state[9+mpc_joint_dim:12+mpc_joint_dim] = euler_derivatives

    # 关节速度 [12+mpc_joint_dim:12+2*mpc_joint_dim]
    for i, joint_name in enumerate(mpc_joint_names):
        try:
            joint_idx = robot_description.get_joint_index(joint_name)
            mpc_state[12+mpc_joint_dim + i] = robot_state.get_joint_velocity(joint_idx)
        except Exception as e:
            print(f"    ⚠ 无法获取关节 {joint_name} 的速度: {e}")

    return mpc_state


def setup_keyboard_listener(vel_cmd):
    """
    设置键盘监听器（在后台线程运行）

    参数:
        vel_cmd: VelocityCommand 对象

    注意:
        在Docker容器中，键盘输入可能需要特殊配置：
        1. 使用Xvfb虚拟显示器: Xvfb :99 -screen 0 1024x768x24 &
        2. 设置DISPLAY环境变量: export DISPLAY=:99
        3. 某些情况下可能需要使用xdotool或其他工具来模拟键盘输入
    """
    if not KEYBOARD_AVAILABLE:
        print("[警告] 键盘输入库不可用，将使用默认速度指令")
        print("[提示] 在Docker容器中，可以安装 pynput 或 keyboard 库: pip install pynput")
        return None

    # 用于调试：记录按键事件
    key_press_count = {'count': 0}

    def on_press(key):
        """键盘按下事件处理"""
        key_press_count['count'] += 1
        try:
            # 使用 pynput
            if hasattr(key, 'char') and key.char is not None:
                char = key.char.lower()
                print(f"[键盘] 检测到按键: '{char}' (总计: {key_press_count['count']})")
                if char == 'w':  # 前进
                    vel_cmd.update_velocity(delta_v_x=vel_cmd.v_x_step)
                    print(f"[键盘] 前进: v_x += {vel_cmd.v_x_step}")
                elif char == 's':  # 后退
                    vel_cmd.update_velocity(delta_v_x=-vel_cmd.v_x_step)
                    print(f"[键盘] 后退: v_x -= {vel_cmd.v_x_step}")
                elif char == 'a':  # 左移
                    vel_cmd.update_velocity(delta_v_y=vel_cmd.v_y_step)
                    print(f"[键盘] 左移: v_y += {vel_cmd.v_y_step}")
                elif char == 'd':  # 右移
                    vel_cmd.update_velocity(delta_v_y=-vel_cmd.v_y_step)
                    print(f"[键盘] 右移: v_y -= {vel_cmd.v_y_step}")
                elif char == 'q':  # 左转
                    vel_cmd.update_velocity(delta_v_yaw=vel_cmd.v_yaw_step)
                    print(f"[键盘] 左转: v_yaw += {vel_cmd.v_yaw_step}")
                elif char == 'e':  # 右转
                    vel_cmd.update_velocity(delta_v_yaw=-vel_cmd.v_yaw_step)
                    print(f"[键盘] 右转: v_yaw -= {vel_cmd.v_yaw_step}")
                elif char == 'r':  # 升高
                    vel_cmd.update_velocity(delta_v_z=vel_cmd.v_z_step)
                    print(f"[键盘] 升高: v_z += {vel_cmd.v_z_step}")
                elif char == 'f':  # 降低
                    vel_cmd.update_velocity(delta_v_z=-vel_cmd.v_z_step)
                    print(f"[键盘] 降低: v_z -= {vel_cmd.v_z_step}")
            else:
                # 特殊键
                key_name = getattr(key, 'name', str(key))
                print(f"[键盘] 检测到特殊键: {key_name} (总计: {key_press_count['count']})")
                if key_name == 'space':  # 停止
                    current_vel = vel_cmd.get_velocities()
                    vel_cmd.update_velocity(
                        delta_v_x=-current_vel[0],
                        delta_v_y=-current_vel[1],
                        delta_v_yaw=-current_vel[3]
                    )
                    print(f"[键盘] 停止: 重置所有速度")
        except AttributeError:
            # 特殊键处理
            if hasattr(key, 'name'):
                if key.name == 'space':  # 停止
                    current_vel = vel_cmd.get_velocities()
                    vel_cmd.update_velocity(
                        delta_v_x=-current_vel[0],
                        delta_v_y=-current_vel[1],
                        delta_v_yaw=-current_vel[3]
                    )
                    print(f"[键盘] 停止: 重置所有速度")
        except Exception as e:
            print(f"[键盘] 处理按键时出错: {e}")

    try:
        # 使用 pynput
        print("[键盘控制] 正在启动键盘监听器...")

        # 检查是否可以使用键盘监听
        try:
            # 尝试创建一个测试监听器来验证能力
            test_listener = keyboard.Listener(on_press=lambda k: None)
            test_listener.start()
            import time
            time.sleep(0.2)
            if test_listener.running:
                test_listener.stop()
                print("[键盘控制] ✓ 键盘监听能力验证成功")
            else:
                print("[键盘控制] ⚠ 键盘监听能力验证失败")
                return None
        except Exception as e:
            print(f"[键盘控制] ⚠ 键盘监听能力验证失败: {e}")
            return None

        # 启动实际监听器
        listener = keyboard.Listener(on_press=on_press)
        listener.start()

        # 等待确认启动
        import time
        time.sleep(0.2)

        if listener.running:
            print("[键盘控制] ✓ 键盘监听器已启动并运行中")
            print("  控制说明:")
            print("    W/S: 前进/后退")
            print("    A/D: 左移/右移")
            print("    Q/E: 左转/右转")
            print("    R/F: 升高/降低")
            print("    Space: 停止")
            print("  提示:")
            print("    - 确保终端窗口处于焦点状态")
            print("    - 如果按键无响应，尝试点击终端窗口后再按键")
            print("    - 可以使用文件控制模式: export USE_FILE_CONTROL=1")
            return listener
        else:
            print("[键盘控制] ⚠ 键盘监听器启动失败（可能无法访问键盘设备）")
            print("  建议: 使用文件控制模式: export USE_FILE_CONTROL=1")
            return None
    except Exception as e:
        import traceback
        print(f"[警告] 无法启动键盘监听器: {e}")
        print("  详细错误:")
        traceback.print_exc()
        print("  可能原因:")
        print("    1. Docker容器无法访问键盘设备（这是正常的）")
        print("    2. 键盘库未正确安装")
        print("    3. 权限不足")
        print("  解决方法:")
        print("    - 使用文件控制模式: export USE_FILE_CONTROL=1")
        print("    - 在宿主机运行脚本（推荐）")
        print("    - 运行诊断: ./check_keyboard.sh")
        return None


def main():
    """主函数：实现完整的 MPC-MRT 控制循环"""

    # ==========================================
    # 1. 初始化：加载模型和启动仿真
    # ==========================================
    print("=" * 60)
    print("G1 人形机器人 MPC-MRT 控制测试 - 稳定行走版本")
    print("=" * 60)

    # 路径配置
    task_file = "/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.info"
    urdf_file = "/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf"
    ref_file = "/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/robot_models/unitree_g1/g1_wb_mpc/config/command/reference.info"
    xml_file = "/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/robot_models/unitree_g1/g1_description/urdf/g1_29dof.xml"

    # 1.1 初始化 WBMpcInterface
    print("\n[1/6] 初始化 WBMpcInterface...")
    try:
        interface = mpc_py.WBMpcInterface(task_file, urdf_file, ref_file, setup_ocp=True)
        interface.setup_mpc()
        state_dim = interface.get_state_dim()
        input_dim = interface.get_input_dim()
        print(f"    ✓ MPC 状态维度: {state_dim}")
        print(f"    ✓ MPC 输入维度: {input_dim}")
    except Exception as e:
        print(f"    ✗ WBMpcInterface 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 1.2 创建目标轨迹计算器
    print("\n[1.2] 创建目标轨迹计算器...")
    try:
        trajectory_calculator = mpc_py.WBMpcTargetTrajectoriesCalculator(ref_file, interface)
        print(f"    ✓ 目标轨迹计算器创建成功")
        print(f"    参考文件: {ref_file}")
    except Exception as e:
        print(f"    ✗ 目标轨迹计算器创建失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 1.3 初始化速度指令管理器
    print("\n[1.3] 初始化速度指令管理器...")

    # 检查是否使用文件控制（环境变量或默认）
    control_file = os.environ.get('VELOCITY_CONTROL_FILE', '/tmp/velocity_control.txt')
    use_file_control = os.environ.get('USE_FILE_CONTROL', '0') == '1'

    if use_file_control:
        vel_cmd = VelocityCommand(control_file=control_file)
        print(f"    ✓ 速度指令管理器创建成功（文件控制模式）")
    else:
        vel_cmd = VelocityCommand()
        print(f"    ✓ 速度指令管理器创建成功（键盘控制模式）")

    print(f"    初始速度指令: [v_x={vel_cmd.commanded_velocities[0]:.2f}, "
          f"v_y={vel_cmd.commanded_velocities[1]:.2f}, "
          f"v_z={vel_cmd.commanded_velocities[2]:.2f}m (高度), "
          f"v_yaw={vel_cmd.commanded_velocities[3]:.2f}]")
    print(f"    注意: v_z 是目标高度（米），不是速度！")

    # 1.4 设置键盘监听器（如果未使用文件控制）
    print("\n[1.4] 设置键盘监听器...")
    keyboard_listener = None
    if not use_file_control:
        keyboard_listener = setup_keyboard_listener(vel_cmd)
    else:
        print("    [跳过] 使用文件控制模式，跳过键盘监听器")
        print(f"    提示: 修改 {control_file} 文件来控制速度")
        print(f"    格式: v_x,v_y,v_z,v_yaw")
        print(f"    示例: echo '0.2,0.0,0.85,0.0' > {control_file}")

    # 1.5 初始化 MujocoSimInterface 并设置初始状态
    print("\n[1.5] 初始化 MujocoSimInterface 并设置初始状态...")
    print("    注意: 初始状态将从 task.info 中的 initialState 配置读取")
    try:
        # 从 MPC 初始状态构建仿真器的初始状态
        # MPC的get_initial_state()会从task.info的initialState配置读取
        init_mpc_state = interface.get_initial_state()
        model_settings = interface.get_model_settings()

        # 提取基座位置 (前3个元素) - 对应task.info中的 p_base_x, p_base_y, p_base_z
        base_pos = init_mpc_state[:3]
        print(f"    基座位置 (来自task.info): [{base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f}]")

        # 提取基座姿态欧拉角 (第3-6个元素: theta_base_z, theta_base_y, theta_base_x)
        base_euler = init_mpc_state[3:6]
        print(f"    基座姿态 (来自task.info): [z={base_euler[0]:.3f}, y={base_euler[1]:.3f}, x={base_euler[2]:.3f}]")

        # 提取关节角度 (从第6个元素开始，共 mpc_joint_dim 个)
        mpc_joint_dim = model_settings.mpc_joint_dim
        mpc_joint_angles = init_mpc_state[6:6+mpc_joint_dim]
        print(f"    关节角度数量: {mpc_joint_dim} (来自task.info)")

        # 获取机器人描述并创建初始状态
        robot_description = mpc_py.RobotDescription(urdf_file)
        init_robot_state = mpc_py.RobotState(robot_description, 2)

        # 设置基座位置（使用task.info中的值）
        init_robot_state.set_root_position(base_pos)

        # 设置基座姿态（从欧拉角转换为四元数）
        # task.info中使用ZYX顺序: theta_base_z, theta_base_y, theta_base_x
        quat = R.from_euler('zyx', base_euler, degrees=False).as_quat()  # 返回 [x, y, z, w]
        quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])  # 转换为 [w, x, y, z]
        init_robot_state.set_root_rotation_quat(quat_wxyz)

        # 设置关节角度（使用task.info中的值）
        mpc_joint_names = model_settings.mpc_model_joint_names
        joint_set_count = 0
        for i, joint_name in enumerate(mpc_joint_names):
            try:
                joint_idx = robot_description.get_joint_index(joint_name)
                init_robot_state.set_joint_position(joint_idx, mpc_joint_angles[i])
                init_robot_state.set_joint_velocity(joint_idx, 0.0)  # 初始速度设为0（task.info中也是0）
                joint_set_count += 1
            except Exception as e:
                print(f"    ⚠ 无法设置关节 {joint_name}: {e}")
        print(f"    成功设置 {joint_set_count}/{mpc_joint_dim} 个关节角度")

        # 设置基座速度为零（task.info中所有速度都是0）
        init_robot_state.set_root_linear_velocity(np.zeros(3))
        init_robot_state.set_root_angular_velocity(np.zeros(3))
        print(f"    基座速度: 零（符合task.info配置）")

        # 创建仿真配置并设置初始状态
        sim_cfg = mpc_py.MujocoSimConfig()
        sim_cfg.scene_path = xml_file
        sim_cfg.dt = 0.0005  # 仿真时间步长
        sim_cfg.render_frequency_hz = 60.0

        # 检查是否启用 headless 模式
        headless_env = os.environ.get('HEADLESS', '0')
        sim_cfg.headless = headless_env == '1'

        # 诊断信息
        display = os.environ.get('DISPLAY', '未设置')
        print(f"    [诊断] DISPLAY={display}")
        print(f"    [诊断] HEADLESS={headless_env} -> headless={sim_cfg.headless}")
        if sim_cfg.headless:
            print(f"    ⚠ 警告: headless 模式已启用，将不会显示 MuJoCo 窗口")
            print(f"    提示: 要显示窗口，请设置 export HEADLESS=0 或取消设置 HEADLESS 环境变量")
        else:
            print(f"    ✓ GUI 模式已启用，将显示 MuJoCo 窗口")

        sim_cfg.verbose = True
        sim_cfg.set_init_state(init_robot_state)  # 设置初始状态

        # 创建仿真器
        sim = mpc_py.MujocoSimInterface(sim_cfg, urdf_file)
        sim.init_sim()

        print(f"    ✓ 仿真器初始化成功")
        print(f"    初始基座位置: [{base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f}]")
        print(f"    初始基座姿态 (欧拉角): [{base_euler[0]:.3f}, {base_euler[1]:.3f}, {base_euler[2]:.3f}]")
        print(f"    MPC 关节数: {mpc_joint_dim}")
    except Exception as e:
        print(f"    ✗ MujocoSimInterface 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 1.6 创建 WBMpcMrtJointController
    print("\n[1.6] 创建 WBMpcMrtJointController...")
    try:
        mpc_frequency = 100.0  # MPC 计算频率 (Hz)
        controller = mpc_py.WBMpcMrtJointController(interface, mpc_frequency)
        print(f"    ✓ 控制器创建成功 (频率: {mpc_frequency} Hz)")
    except Exception as e:
        print(f"    ✗ WBMpcMrtJointController 创建失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # ==========================================
    # 2. 感知：获取初始状态并启动 MPC 线程
    # ==========================================
    print("\n[2/6] 启动 MPC 后台线程...")
    try:
        # 获取初始机器人状态
        initial_robot_state = sim.get_robot_state()

        # 启动 MPC 后台线程
        controller.start_mpc_thread(initial_robot_state)
        print(f"    ✓ MPC 线程已启动")

        # 等待 MPC 策略就绪
        print(f"    等待 MPC 策略就绪...", end='', flush=True)
        max_wait_time = 10.0  # 最大等待时间（秒）
        wait_start = time.time()
        while not controller.ready():
            if time.time() - wait_start > max_wait_time:
                print(f"\n    ✗ 超时：MPC 策略未能在 {max_wait_time} 秒内就绪")
                return
            time.sleep(0.1)
            print('.', end='', flush=True)
        print(f" 就绪！")

    except Exception as e:
        print(f"    ✗ 启动 MPC 线程失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # ==========================================
    # 3. 设置初始目标轨迹
    # ==========================================
    print("\n[3/6] 设置初始目标轨迹...")
    try:
        # 获取初始 MPC 状态
        initial_mpc_state = robot_state_to_mpc_state(initial_robot_state, interface, robot_description)

        # 使用初始速度指令生成目标轨迹
        initial_velocities = vel_cmd.get_velocities()

        # 确保初始高度设置合理（使用task.info中的初始高度或当前基座高度）
        current_base_pos = initial_robot_state.get_root_position()
        current_height = current_base_pos[2]
        # task.info中p_base_z=0.8m，reference.info中defaultBaseHeight=0.7925m
        # 如果速度指令中的高度不合理，使用当前基座高度（应该接近0.8m）
        if initial_velocities[2] < 0.7 or initial_velocities[2] > 1.0:
            initial_velocities[2] = max(0.7, min(1.0, current_height))
            print(f"    [调整] 初始高度设置为当前基座高度: {initial_velocities[2]:.3f}m")
        else:
            print(f"    [确认] 初始高度: {initial_velocities[2]:.3f}m (当前基座高度: {current_height:.3f}m)")

        print(f"    初始速度指令: [v_x={initial_velocities[0]:.2f}, v_y={initial_velocities[1]:.2f}, "
              f"v_z={initial_velocities[2]:.2f}m, v_yaw={initial_velocities[3]:.2f}]")

        initial_target_traj = trajectory_calculator.commanded_velocity_to_target_trajectories(
            commanded_velocities=initial_velocities,
            init_time=initial_robot_state.get_time(),
            init_state=initial_mpc_state
        )

        # 将目标轨迹写入 MPC 缓冲区
        interface.set_target_trajectories(initial_target_traj)
        print(f"    ✓ 初始目标轨迹已设置")
    except Exception as e:
        print(f"    ✗ 设置初始目标轨迹失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # ==========================================
    # 4-5. 主控制循环：感知 -> 决策 -> 执行 -> 驱动
    # ==========================================
    print("\n[4-5/6] 启动主控制循环...")
    print("-" * 60)

    # 控制循环参数
    control_frequency = 500.0  # 控制频率 (Hz)
    control_dt = 1.0 / control_frequency

    # 轨迹更新频率（降低频率以减少计算负担，避免MPC求解器过载）
    # 注意：MPC求解器需要时间收敛，频繁更新轨迹会导致求解器无法收敛
    # task.info中sqpIteration=2，每次MPC计算需要约30ms，建议更新频率不超过10Hz
    trajectory_update_frequency = 3.0  # Hz (进一步降低到3Hz，给MPC更多时间收敛)
    trajectory_update_dt = 1.0 / trajectory_update_frequency
    last_trajectory_update_time = time.time()
    last_velocities = None  # 用于检测速度指令是否变化

    print(f"    轨迹更新频率: {trajectory_update_frequency} Hz (每 {trajectory_update_dt:.2f} 秒更新一次)")
    print(f"    注意: MPC求解器需要时间收敛，频繁更新可能导致求解失败")

    frame_count = 0
    last_print_time = time.time()
    start_time = time.time()
    initial_pos = None  # 用于跟踪初始位置

    try:
        while True:
            loop_start = time.time()

            # ===== 2. 感知：从仿真器获取 RobotState =====
            sim.update_interface_state_from_robot()
            robot_state = sim.get_robot_state()
            current_time = robot_state.get_time()

            # ===== 3. 决策：根据键盘输入更新目标轨迹 =====
            # MPC 计算在后台线程自动进行，这里只需要检查是否就绪
            if not controller.ready():
                print(f"\n    ⚠ MPC 策略未就绪，跳过此帧")
                time.sleep(control_dt)
                continue

            # 定期更新目标轨迹（根据当前速度指令）
            # 只在速度指令变化或达到更新间隔时才更新，避免频繁更新导致MPC无法收敛
            current_real_time = time.time()
            current_velocities = vel_cmd.get_velocities()

            # 检查速度指令是否变化（使用小的阈值避免浮点误差）
            velocities_changed = False
            if last_velocities is None:
                velocities_changed = True
            else:
                vel_diff = np.abs(current_velocities - last_velocities)
                if np.any(vel_diff > 0.01):  # 速度变化超过1cm/s或0.01rad/s
                    velocities_changed = True

            # 更新条件：速度变化 或 达到更新间隔
            # 注意：为了给MPC更多时间收敛，我们降低更新频率
            should_update = velocities_changed or (current_real_time - last_trajectory_update_time >= trajectory_update_dt)

            if should_update:
                try:
                    # 将 RobotState 转换为 MPC 状态向量
                    current_mpc_state = robot_state_to_mpc_state(robot_state, interface, robot_description)

                    # 使用速度指令生成新的目标轨迹
                    # 注意：commanded_velocity_to_target_trajectories 会自动：
                    # 1. 将速度指令积分生成期望基座位置和速度
                    # 2. 摆动腿规划器会根据当前相位规划脚在空中的期望轨迹
                    new_target_traj = trajectory_calculator.commanded_velocity_to_target_trajectories(
                        commanded_velocities=current_velocities,
                        init_time=current_time,
                        init_state=current_mpc_state
                    )

                    # 检查目标轨迹是否有效
                    if new_target_traj.size() == 0:
                        print(f"\n    ⚠ 警告: 生成的目标轨迹为空")
                    else:
                        # 获取轨迹的第一个和最后一个状态，用于验证
                        first_state = new_target_traj.state_trajectory[0]
                        last_state = new_target_traj.state_trajectory[-1]
                        first_time = new_target_traj.time_trajectory[0]
                        last_time = new_target_traj.time_trajectory[-1]

                        # 提取期望基座位置和速度（用于调试）
                        model_settings = interface.get_model_settings()
                        mpc_joint_dim = model_settings.mpc_joint_dim
                        desired_base_pos_start = first_state[:3]
                        desired_base_pos_end = last_state[:3]
                        # 基座速度在状态向量中的位置: [base_pos(3), base_euler(3), joints(n), base_vel(3), ...]
                        vel_start_idx = 6 + mpc_joint_dim
                        vel_end_idx = vel_start_idx + 3
                        if len(first_state) > vel_end_idx:
                            desired_base_vel_start = first_state[vel_start_idx:vel_end_idx]
                        else:
                            desired_base_vel_start = np.zeros(3)

                        # 将新轨迹写入 MPC 的线程安全缓冲区
                        interface.set_target_trajectories(new_target_traj)

                        if velocities_changed:
                            print(f"\n[轨迹更新] ✓ 速度指令变化: v_x={current_velocities[0]:.2f}, "
                                  f"v_y={current_velocities[1]:.2f}, v_z={current_velocities[2]:.2f}m, "
                                  f"v_yaw={current_velocities[3]:.2f}")
                            print(f"           期望基座位置: [{desired_base_pos_start[0]:.3f}, "
                                  f"{desired_base_pos_start[1]:.3f}, {desired_base_pos_start[2]:.3f}] -> "
                                  f"[{desired_base_pos_end[0]:.3f}, {desired_base_pos_end[1]:.3f}, "
                                  f"{desired_base_pos_end[2]:.3f}]")
                            print(f"           期望基座速度: [{desired_base_vel_start[0]:.3f}, "
                                  f"{desired_base_vel_start[1]:.3f}, {desired_base_vel_start[2]:.3f}]")
                            print(f"           轨迹时间范围: {first_time:.3f}s -> {last_time:.3f}s "
                                  f"(时长: {last_time-first_time:.3f}s)")

                    last_trajectory_update_time = current_real_time
                    last_velocities = current_velocities.copy()
                except Exception as e:
                    print(f"\n    ⚠ 更新目标轨迹失败: {e}")
                    import traceback
                    traceback.print_exc()

            # ===== 4. 执行：WBMpcMrtJointController 将 MPC 结果转换为 RobotJointAction =====
            joint_actions = sim.get_robot_joint_action()
            controller.compute_joint_control_action(current_time, robot_state, joint_actions)

            # ===== 5. 驱动：将动作作用于 MuJoCo 仿真器 =====
            sim.apply_joint_action()
            sim.simulation_step()

            # ===== 调试信息打印 =====
            frame_count += 1
            if current_real_time - last_print_time >= 0.5:  # 每 0.5 秒打印一次
                root_pos = robot_state.get_root_position()
                root_quat = robot_state.get_root_rotation_quat()
                current_velocities_display = vel_cmd.get_velocities()

                height = root_pos[2]

                # 计算基座速度（用于验证MPC是否响应速度指令）
                base_vel_local = robot_state.get_root_linear_velocity()
                base_vel_world_norm = np.linalg.norm(base_vel_local)  # 基座速度大小

                # 计算基座位置变化（用于验证是否在移动）
                if initial_pos is None:
                    initial_pos = root_pos.copy()
                    pos_change = 0.0
                else:
                    pos_change = np.linalg.norm(root_pos[:2] - initial_pos[:2])  # XY平面移动距离

                # 打印状态信息
                elapsed_time = current_real_time - start_time
                print(f"\nFrame: {frame_count:6d} | "
                      f"Time: {current_time:6.2f}s | "
                      f"Height: {height:.3f}m | "
                      f"Pos: [{root_pos[0]:.3f}, {root_pos[1]:.3f}, {root_pos[2]:.3f}] | "
                      f"移动距离: {pos_change:.3f}m | "
                      f"Vel: {base_vel_world_norm:.3f}m/s | "
                      f"Cmd: [v_x={current_velocities_display[0]:.2f}, v_y={current_velocities_display[1]:.2f}, "
                      f"v_z={current_velocities_display[2]:.2f}m, v_yaw={current_velocities_display[3]:.2f}] | "
                      f"MPC: {'✓' if controller.ready() else '✗'}",
                      end='')
                last_print_time = current_real_time

            # ===== 频率控制 =====
            elapsed = time.time() - loop_start
            if elapsed < control_dt:
                time.sleep(control_dt - elapsed)
            else:
                # 控制循环运行过慢
                if frame_count % 100 == 0:
                    print(f"\n    ⚠ 控制循环运行缓慢: 期望 {control_dt*1000:.1f}ms, 实际 {elapsed*1000:.1f}ms")

    except KeyboardInterrupt:
        print("\n\n    [INFO] 用户中断仿真")
    except Exception as e:
        print(f"\n\n    [错误] 控制循环异常: {e}")
        import traceback
        traceback.print_exc()

    # ==========================================
    # 6. 清理资源
    # ==========================================
    print("\n[6/6] 清理资源...")
    if keyboard_listener is not None:
        try:
            keyboard_listener.stop()
            print("    ✓ 键盘监听器已停止")
        except:
            pass

    print("\n" + "-" * 60)
    print("[完成] 仿真结束")
    print(f"总帧数: {frame_count}")
    print(f"总时间: {time.time() - start_time:.2f}s")
    if frame_count > 0:
        print(f"平均频率: {frame_count / (time.time() - start_time):.1f} Hz")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"\n[错误] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
