#!/usr/bin/env python3
"""
G1 人形机器人 MPC-MRT 行走控制测试脚本

与 C++ WBMpcRobotSim 一致：使用 ProceduralMpcMotionManager 做速度/高度限幅、滤波与步态联动。
1. 用户指令流：从 velocity_control 文件中读取 [v_x, v_y, desired_pelvis_height, v_yaw]
2. 程序化行走管理：ProceduralMpcMotionManager 根据指令更新 TargetTrajectories 与 GaitSchedule
3. MPC 求解器：在后台线程中根据最新参考求解
4. 高频 MRT 关节控制：在仿真主循环中插值 MPC 结果驱动关节
"""

import sys
import os
import time
import numpy as np

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

# 先预加载系统 libglfw 与 libGLEW（RTLD_GLOBAL），避免 humanoid_wb_mpc_py 加载时出现 undefined symbol（glfwSetScrollCallback / glewInit）
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
    print("请确认已在工作空间中成功编译并安装 humanoid_wb_mpc 的 Python 绑定。")
    sys.exit(1)


# ==========================
# 用户速度指令文件
# ==========================
VELOCITY_CONTROL_FILE = os.environ.get("VELOCITY_CONTROL_FILE", "/tmp/velocity_control.txt")


def _read_latest_velocity_command(last_cmd: np.ndarray) -> np.ndarray:
    """
    从速度控制文件中读取最新一行指令。

    文件格式由 interactive_control.py 维护，每行:
        v_x, v_y, v_z, v_yaw

    若读取失败，则返回传入的 last_cmd（保持上一次指令）。
    """
    if not os.path.exists(VELOCITY_CONTROL_FILE):
        return last_cmd

    try:
        with open(VELOCITY_CONTROL_FILE, "r") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        if not lines:
            return last_cmd

        parts = lines[-1].split(",")
        if len(parts) != 4:
            print(f"[警告] 控制文件格式错误（期望4列），忽略该行: {lines[-1]}")
            return last_cmd

        values = [float(p) for p in parts]
        return np.array(values, dtype=float)
    except Exception as e:
        print(f"[警告] 读取速度控制文件失败: {e}")
        return last_cmd


def main():
    """主函数：实现带行走的 MPC-MRT 控制循环"""

    # ==========================================
    # 1. 初始化：加载模型和启动仿真
    # ==========================================
    print("=" * 60)
    print("G1 人形机器人 MPC-MRT 行走控制测试")
    print("=" * 60)

    # 路径配置
    task_file = (
        "/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.info"
    )
    urdf_file = (
        "/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf"
    )
    ref_file = (
        "/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/robot_models/unitree_g1/g1_wb_mpc/config/command/reference.info"
    )
    xml_file = (
        "/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/robot_models/unitree_g1/g1_description/urdf/g1_29dof.xml"
    )
    # 与 ROS2 启动脚本保持一致的 gait.info 路径
    gait_file = (
        "/wb_humanoid_mpc_ws/install/humanoid_common_mpc/share/humanoid_common_mpc/config/command/gait.info"
    )

    # 1.1 初始化 WBMpcInterface
    print("\n[1/5] 初始化 WBMpcInterface...")
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

    # 1.2 创建 ProceduralMpcMotionManager（与 C++ WBMpcRobotSim 一致：限幅、滤波、步态联动）
    print("\n[1.2] 创建 ProceduralMpcMotionManager 并设置初始参考...")
    try:
        proc_mgr = mpc_py.ProceduralMpcMotionManager(gait_file, ref_file, interface)
        init_mpc_state = np.array(interface.get_initial_state(), dtype=float)
        initial_base_height = float(init_mpc_state[2])
        commanded_vel = np.array([0.0, 0.0, initial_base_height, 0.0], dtype=float)
        commanded_vel = _read_latest_velocity_command(commanded_vel)
        print(
            "    初始速度/高度指令: "
            f"v_x={commanded_vel[0]:.3f}, v_y={commanded_vel[1]:.3f}, "
            f"base_z={commanded_vel[2]:.3f}, v_yaw={commanded_vel[3]:.3f}"
        )
        proc_mgr.set_velocity_command(commanded_vel)
        # 参考由 MPC 求解器每步在 preSolverRun 中更新（manager 已挂为 synchronized module），无需在此调用 update_references
        print("    ✓ 已设置初始速度/高度指令；参考将在每次 MPC 求解前由求解器自动更新")
    except Exception as e:
        print(f"    ✗ 创建或设置 ProceduralMpcMotionManager 失败: {e}")
        import traceback

        traceback.print_exc()
        return

    # 可选：获取步态参考管理器（用于调试接触状态与步态相位）
    try:
        ref_manager = interface.get_switched_model_reference_manager_ptr()
        swing_planner = ref_manager.get_swing_trajectory_planner()
        print("    ✓ 已获取 SwitchedModelReferenceManager 与 SwingTrajectoryPlanner")
    except Exception as e:
        ref_manager = None
        swing_planner = None
        print(f"    ⚠ 无法获取步态参考管理器（可选功能）: {e}")

    # 1.3 初始化 MujocoSimInterface 并设置初始状态
    print("\n[1.3] 初始化 MujocoSimInterface 并设置初始状态...")
    try:
        init_mpc_state = interface.get_initial_state()
        model_settings = interface.get_model_settings()

        # 提取基座位置 (前3个元素)
        base_pos = init_mpc_state[:3]

        # 提取基座姿态欧拉角 (第3-6个元素: roll, pitch, yaw)
        base_euler = init_mpc_state[3:6]

        # 提取关节角度 (从第6个元素开始，共 mpc_joint_dim 个)
        mpc_joint_dim = model_settings.mpc_joint_dim
        mpc_joint_angles = init_mpc_state[6 : 6 + mpc_joint_dim]

        # 获取机器人描述并创建初始状态
        robot_description = mpc_py.RobotDescription(urdf_file)
        init_robot_state = mpc_py.RobotState(robot_description, 2)

        # 设置基座位置
        init_robot_state.set_root_position(base_pos)

        # 设置基座姿态（从欧拉角转换为四元数）: [w, x, y, z]
        from scipy.spatial.transform import Rotation as R

        quat_xyzw = R.from_euler("zyx", base_euler, degrees=False).as_quat()  # [x, y, z, w]
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        init_robot_state.set_root_rotation_quat(quat_wxyz)

        # 设置关节角度
        mpc_joint_names = model_settings.mpc_model_joint_names
        for i, joint_name in enumerate(mpc_joint_names):
            try:
                joint_idx = robot_description.get_joint_index(joint_name)
                init_robot_state.set_joint_position(joint_idx, mpc_joint_angles[i])
                init_robot_state.set_joint_velocity(joint_idx, 0.0)
            except Exception as e:
                print(f"    ⚠ 无法设置关节 {joint_name}: {e}")

        # 设置基座速度为零
        init_robot_state.set_root_linear_velocity(np.zeros(3))
        init_robot_state.set_root_angular_velocity(np.zeros(3))

        # 创建仿真配置并设置初始状态
        sim_cfg = mpc_py.MujocoSimConfig()
        sim_cfg.scene_path = xml_file
        sim_cfg.dt = 0.0005  # 仿真时间步长
        sim_cfg.render_frequency_hz = 60.0
        sim_cfg.headless = os.environ.get("HEADLESS", "0") == "1"
        sim_cfg.verbose = True
        sim_cfg.set_init_state(init_robot_state)

        # 创建仿真器
        sim = mpc_py.MujocoSimInterface(sim_cfg, urdf_file)
        sim.init_sim()

        print("    ✓ 仿真器初始化成功")
        print(
            f"    初始基座位置: [{base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f}] "
            f"姿态(euler): [{base_euler[0]:.3f}, {base_euler[1]:.3f}, {base_euler[2]:.3f}]"
        )
        print(f"    MPC 关节数: {mpc_joint_dim}")
    except Exception as e:
        print(f"    ✗ MujocoSimInterface 初始化失败: {e}")
        import traceback

        traceback.print_exc()
        return

    # 1.4 创建 WBMpcMrtJointController
    print("\n[1.4] 创建 WBMpcMrtJointController...")
    try:
        mpc_frequency = 100.0  # MPC 计算频率 (Hz)
        controller = mpc_py.WBMpcMrtJointController(interface, mpc_frequency)
        print(f"    ✓ 控制器创建成功 (频率: {mpc_frequency} Hz)")
    except Exception as e:
        print(f"    ✗ WBMpcMrtJointController 创建失败: {e}")
        import traceback

        traceback.print_exc()
        return

    # 1.5 获取 RobotJointAction 容器（从仿真器）
    print("\n[1.5] 获取 RobotJointAction 容器...")
    try:
        joint_actions = sim.get_robot_joint_action()
        print(f"    ✓ 关节动作容器获取成功 (关节数: {len(joint_actions)})")
    except Exception as e:
        print(f"    ✗ 获取关节动作容器失败: {e}")
        import traceback

        traceback.print_exc()
        return

    # ==========================================
    # 2. 感知：获取初始状态并启动 MPC 线程
    # ==========================================
    print("\n[2/5] 启动 MPC 后台线程...")
    try:
        # 获取初始机器人状态
        initial_robot_state = sim.get_robot_state()

        # 启动 MPC 后台线程
        controller.start_mpc_thread(initial_robot_state)
        print("    ✓ MPC 线程已启动")

        # 等待 MPC 策略就绪
        print("    等待 MPC 策略就绪...", end="", flush=True)
        max_wait_time = 10.0
        wait_start = time.time()
        while not controller.ready():
            if time.time() - wait_start > max_wait_time:
                print(f"\n    ✗ 超时：MPC 策略未能在 {max_wait_time} 秒内就绪")
                return
            time.sleep(0.1)
            print(".", end="", flush=True)
        print(" 就绪！")
    except Exception as e:
        print(f"    ✗ 启动 MPC 线程失败: {e}")
        import traceback

        traceback.print_exc()
        return

    # ==========================================
    # 3. 决策：可选 RL 权重调整模块
    # ==========================================
    print("\n[3/5] 设置 RL 权重调整模块（可选）...")
    weight_adjustment_module = None
    try:
        weight_adjustment_module = interface.get_weight_adjustment_module()
        print("    ✓ MpcWeightAdjustmentModule 已获取")
        print("    提示：可通过 set_residual_weights() 注入 RL 策略权重")
    except Exception as e:
        print(f"    ⚠ 无法获取权重调整模块（可选）: {e}")

    # ==========================================
    # 4-5. 主控制循环：感知 -> 决策 -> 执行 -> 驱动
    # ==========================================
    print("\n[4-5/5] 启动主控制循环（含行走指令与步态调度）...")
    print("-" * 60)

    control_frequency = 500.0  # 控制频率 (Hz)
    control_dt = 1.0 / control_frequency

    # 速度指令与参考更新（与 C++ 一致：ProceduralMpcMotionManager 内部做限幅、滤波与步态切换）
    commanded_vel = _read_latest_velocity_command(commanded_vel)
    last_cmd_update_time = 0.0
    command_update_period = 0.05  # 每 0.05s 从文件读取并写入速度/高度指令（约 20Hz）

    frame_count = 0
    last_print_time = time.time()
    last_gait_print_time = 0.0
    start_time = time.time()

    try:
        while True:
            loop_start_wall = time.time()

            # ===== 2. 感知：从仿真器获取 RobotState =====
            sim.update_interface_state_from_robot()
            robot_state = sim.get_robot_state()
            current_time = robot_state.get_time()

            # ===== 2.1 读取速度/高度指令并写入 manager（参考由求解器每步 preSolverRun 自动更新） =====
            if current_time - last_cmd_update_time >= command_update_period:
                new_cmd = _read_latest_velocity_command(commanded_vel)
                if not np.allclose(new_cmd, commanded_vel):
                    print(
                        f"\n    [指令变化 @ {current_time:.2f}s] "
                        f"v_x={new_cmd[0]:.2f}, v_y={new_cmd[1]:.2f}, "
                        f"base_z={new_cmd[2]:.2f}, v_yaw={new_cmd[3]:.2f}"
                    )
                commanded_vel = new_cmd
                try:
                    proc_mgr.set_velocity_command(commanded_vel)
                except Exception as e:
                    print(f"\n    ✗ 设置速度/高度指令失败: {e}")
                last_cmd_update_time = current_time

            # ===== 3. 决策：检查 MPC 是否就绪 =====
            if not controller.ready():
                print("\n    ⚠ MPC 策略未就绪，跳过此帧")
                time.sleep(control_dt)
                continue

            # ===== 4. 执行：MRT 控制器计算关节动作 =====
            joint_actions = sim.get_robot_joint_action()
            controller.compute_joint_control_action(current_time, robot_state, joint_actions)

            # ===== 5. 驱动：动作作用到 MuJoCo 仿真器 =====
            sim.apply_joint_action()
            sim.simulation_step()

            # ===== 调试信息打印 =====
            frame_count += 1
            now_wall = time.time()

            if now_wall - last_print_time >= 0.5:
                root_pos = robot_state.get_root_position()
                root_quat = robot_state.get_root_rotation_quat()
                height = root_pos[2]

                gait_str = ""
                if ref_manager is not None:
                    try:
                        contact_flags = ref_manager.get_contact_flags(current_time)
                        # contact_flags: [左脚, 右脚]，True 表示着地
                        left_contact = "S" if contact_flags[0] else "W"
                        right_contact = "S" if contact_flags[1] else "W"
                        gait_str = f" | Gait(L,R)=[{left_contact},{right_contact}]"
                    except Exception:
                        gait_str = ""

                elapsed_time = now_wall - start_time
                print(
                    f"Frame: {frame_count:6d} | "
                    f"SimTime: {current_time:6.2f}s | "
                    f"Height: {height:.3f}m | "
                    f"Pos: [{root_pos[0]:.3f}, {root_pos[1]:.3f}, {root_pos[2]:.3f}] | "
                    f"Quat: [w={root_quat[0]:.3f}, x={root_quat[1]:.3f}, y={root_quat[2]:.3f}, z={root_quat[3]:.3f}]"
                    f"{gait_str}",
                    end="\r",
                )
                last_print_time = now_wall

            # 可选：定期打印更详细的步态相位 / 摆动脚高度
            if swing_planner is not None and current_time - last_gait_print_time >= 0.5:
                try:
                    # 0: 左脚, 1: 右脚
                    z_l = swing_planner.get_z_position_constraint(0, current_time)
                    z_r = swing_planner.get_z_position_constraint(1, current_time)
                    print(
                        f"\n    [步态信息 @ {current_time:.2f}s] "
                        f"swing_z_L={z_l:.3f}, swing_z_R={z_r:.3f}"
                    )
                except Exception:
                    pass
                last_gait_print_time = current_time

            # ===== 频率控制 =====
            elapsed = time.time() - loop_start_wall
            if elapsed < control_dt:
                time.sleep(control_dt - elapsed)
            else:
                if frame_count % 100 == 0:
                    print(
                        f"\n    ⚠ 控制循环运行缓慢: 期望 {control_dt*1000:.1f}ms, 实际 {elapsed*1000:.1f}ms"
                    )

    except KeyboardInterrupt:
        print("\n\n    [INFO] 用户中断仿真")
    except Exception as e:
        print(f"\n\n    [错误] 控制循环异常: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "-" * 60)
    print("[完成] 行走仿真结束")
    print(f"总帧数: {frame_count}")
    total_time = time.time() - start_time
    print(f"总时间: {total_time:.2f}s")
    if total_time > 0.0:
        print(f"平均频率: {frame_count / total_time:.1f} Hz")


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

