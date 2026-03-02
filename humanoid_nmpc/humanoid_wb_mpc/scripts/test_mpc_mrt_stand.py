"""
G1 人形机器人 MPC-MRT 控制测试脚本

实现完整的数据流：
1. 初始化：通过 WBMpcInterface 加载模型，通过 MujocoSimInterface 启动仿真
2. 感知：从仿真器获取 RobotState
3. 决策：WBMpcInterface 在后台线程计算最优路径；若有 RL 策略，通过 MpcWeightAdjustmentModule 注入权重
4. 执行：WBMpcMrtJointController 将 MPC 结果转换为 RobotJointAction
5. 驱动：将动作作用于 MuJoCo 里的机器人，循环往复
"""

import sys
import os
import time
import numpy as np

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
    print(f"请确保已编译并安装 Python 绑定模块")
    sys.exit(1)


def main():
    """主函数：实现完整的 MPC-MRT 控制循环"""

    # ==========================================
    # 1. 初始化：加载模型和启动仿真
    # ==========================================
    print("=" * 60)
    print("G1 人形机器人 MPC-MRT 控制测试")
    print("=" * 60)

    # 路径配置
    task_file = "/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.info"
    urdf_file = "/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf"
    ref_file = "/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/robot_models/unitree_g1/g1_wb_mpc/config/command/reference.info"
    xml_file = "/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/robot_models/unitree_g1/g1_description/urdf/g1_29dof.xml"

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

    # 1.2 设置目标轨迹（初始状态）
    print("\n[1.2] 设置目标轨迹...")
    try:
        initial_state = interface.get_initial_state()
        interface.set_target_state(initial_state)
        print(f"    ✓ 目标轨迹已设置")
    except Exception as e:
        print(f"    ✗ 设置目标轨迹失败: {e}")
        return

    # 1.3 初始化 MujocoSimInterface 并设置初始状态
    print("\n[1.3] 初始化 MujocoSimInterface 并设置初始状态...")
    try:
        # 从 MPC 初始状态构建仿真器的初始状态
        # MPC 状态向量结构: [base_pos(3), base_euler(3), joint_angles(n), base_vel(3), base_ang_vel(3), joint_vels(n)]
        init_mpc_state = interface.get_initial_state()
        model_settings = interface.get_model_settings()

        # 提取基座位置 (前3个元素)
        base_pos = init_mpc_state[:3]

        # 提取基座姿态欧拉角 (第3-6个元素: roll, pitch, yaw)
        base_euler = init_mpc_state[3:6]

        # 提取关节角度 (从第6个元素开始，共 mpc_joint_dim 个)
        mpc_joint_dim = model_settings.mpc_joint_dim
        mpc_joint_angles = init_mpc_state[6:6+mpc_joint_dim]

        # 获取机器人描述并创建初始状态
        robot_description = mpc_py.RobotDescription(urdf_file)
        init_robot_state = mpc_py.RobotState(robot_description, 2)

        # 设置基座位置
        init_robot_state.set_root_position(base_pos)

        # 设置基座姿态（从欧拉角转换为四元数）
        # 注意：需要将欧拉角转换为四元数 [w, x, y, z]
        from scipy.spatial.transform import Rotation as R
        quat = R.from_euler('zyx', base_euler, degrees=False).as_quat()  # 返回 [x, y, z, w]
        quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])  # 转换为 [w, x, y, z]
        init_robot_state.set_root_rotation_quat(quat_wxyz)

        # 设置关节角度
        mpc_joint_names = model_settings.mpc_model_joint_names
        for i, joint_name in enumerate(mpc_joint_names):
            try:
                joint_idx = robot_description.get_joint_index(joint_name)
                init_robot_state.set_joint_position(joint_idx, mpc_joint_angles[i])
                init_robot_state.set_joint_velocity(joint_idx, 0.0)  # 初始速度设为0
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
        sim_cfg.headless = os.environ.get('HEADLESS', '0') == '1'
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
    # 3. 决策：可选 RL 权重调整模块
    # ==========================================
    print("\n[3/5] 设置 RL 权重调整模块（可选）...")
    weight_adjustment_module = None
    try:
        weight_adjustment_module = interface.get_weight_adjustment_module()
        print(f"    ✓ MpcWeightAdjustmentModule 已获取")
        print(f"    提示：可通过 set_residual_weights() 注入 RL 策略权重")
    except Exception as e:
        print(f"    ⚠ 无法获取权重调整模块（可选）: {e}")
        # 这不是致命错误，继续执行

    # ==========================================
    # 4-5. 主控制循环：感知 -> 决策 -> 执行 -> 驱动
    # ==========================================
    print("\n[4-5/5] 启动主控制循环...")
    print("-" * 60)

    # 控制循环参数
    control_frequency = 500.0  # 控制频率 (Hz)
    control_dt = 1.0 / control_frequency

    frame_count = 0
    last_print_time = time.time()
    start_time = time.time()

    try:
        while True:
            loop_start = time.time()

            # ===== 2. 感知：从仿真器获取 RobotState =====
            # 更新接口状态（从线程安全状态复制到可访问状态）
            sim.update_interface_state_from_robot()
            robot_state = sim.get_robot_state()
            current_time = robot_state.get_time()

            # ===== 3. 决策：MPC 在后台线程计算最优路径 =====
            # MPC 计算在后台线程自动进行，这里只需要检查是否就绪
            if not controller.ready():
                print(f"\n    ⚠ MPC 策略未就绪，跳过此帧")
                time.sleep(control_dt)
                continue

            # 可选：通过 RL 策略注入权重
            # if weight_adjustment_module is not None:
            #     # 示例：假设 RL 策略输出 58 个权重值
            #     # rl_weights = np.random.randn(58)  # 替换为实际的 RL 策略输出
            #     # weight_adjustment_module.set_residual_weights(rl_weights.tolist())
            #     pass

            # ===== 4. 执行：WBMpcMrtJointController 将 MPC 结果转换为 RobotJointAction =====
            # 获取关节动作容器（从仿真器）
            joint_actions = sim.get_robot_joint_action()
            # 计算并填充关节控制动作
            # 注意：compute_joint_control_action 会修改 joint_actions 容器
            controller.compute_joint_control_action(current_time, robot_state, joint_actions)

            # ===== 5. 驱动：将动作作用于 MuJoCo 仿真器 =====
            # 将动作应用到仿真器（设置到线程安全的动作缓冲区）
            sim.apply_joint_action()
            # 执行仿真步进（会自动从线程安全缓冲区读取动作，计算力矩，推进仿真，更新状态）
            sim.simulation_step()

            # ===== 调试信息打印 =====
            frame_count += 1
            current_real_time = time.time()
            if current_real_time - last_print_time >= 0.5:  # 每 0.5 秒打印一次
                root_pos = robot_state.get_root_position()
                root_quat = robot_state.get_root_rotation_quat()  # [w, x, y, z] 格式

                # 计算基座高度和姿态
                height = root_pos[2]

                # 打印状态信息
                elapsed_time = current_real_time - start_time
                print(f"Frame: {frame_count:6d} | "
                      f"Time: {current_time:6.2f}s | "
                      f"Height: {height:.3f}m | "
                      f"Pos: [{root_pos[0]:.3f}, {root_pos[1]:.3f}, {root_pos[2]:.3f}] | "
                      f"Quat: [w={root_quat[0]:.3f}, x={root_quat[1]:.3f}, y={root_quat[2]:.3f}, z={root_quat[3]:.3f}]",
                      end='\r')
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

    print("\n" + "-" * 60)
    print("[完成] 仿真结束")
    print(f"总帧数: {frame_count}")
    print(f"总时间: {time.time() - start_time:.2f}s")
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