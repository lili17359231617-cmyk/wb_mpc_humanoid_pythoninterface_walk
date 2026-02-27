#!/usr/bin/env python3
"""
交互式速度控制脚本
用于在Docker容器中通过命令行交互控制机器人速度
"""

import sys
import os
import time
import select
import termios
import tty

CONTROL_FILE = os.environ.get('VELOCITY_CONTROL_FILE', '/tmp/velocity_control.txt')

# 当前速度（初始高度使用task.info中的p_base_z=0.8m）
current_vel = [0.0, 0.0, 0.8, 0.0]  # v_x, v_y, v_z, v_yaw

# 速度增量
v_x_step = 0.1
v_y_step = 0.1
v_z_step = 0.02
v_yaw_step = 0.2

# 速度限制
v_x_min, v_x_max = -0.5, 0.5
v_y_min, v_y_max = -0.3, 0.3
v_z_min, v_z_max = 0.7, 1.0
v_yaw_min, v_yaw_max = -0.5, 0.5

def update_velocity(delta_v_x=0.0, delta_v_y=0.0, delta_v_z=0.0, delta_v_yaw=0.0):
    """更新速度并写入文件"""
    global current_vel

    current_vel[0] = max(v_x_min, min(v_x_max, current_vel[0] + delta_v_x))
    current_vel[1] = max(v_y_min, min(v_y_max, current_vel[1] + delta_v_y))
    current_vel[2] = max(v_z_min, min(v_z_max, current_vel[2] + delta_v_z))
    current_vel[3] = max(v_yaw_min, min(v_yaw_max, current_vel[3] + delta_v_yaw))

    # 写入控制文件
    with open(CONTROL_FILE, 'a') as f:
        f.write(f"{current_vel[0]},{current_vel[1]},{current_vel[2]},{current_vel[3]}\n")

    return current_vel.copy()

def print_status():
    """打印当前状态"""
    print(f"\r速度: v_x={current_vel[0]:.2f}, v_y={current_vel[1]:.2f}, "
          f"v_z={current_vel[2]:.2f}, v_yaw={current_vel[3]:.2f}  ", end='', flush=True)

def print_help():
    """打印帮助信息"""
    print("\n" + "=" * 60)
    print("交互式速度控制")
    print("=" * 60)
    print("\n控制命令:")
    print("  w/W - 前进 (v_x += 0.1)")
    print("  s/S - 后退 (v_x -= 0.1)")
    print("  a/A - 左移 (v_y += 0.1)")
    print("  d/D - 右移 (v_y -= 0.1)")
    print("  q/Q - 左转 (v_yaw += 0.2)")
    print("  e/E - 右转 (v_yaw -= 0.2)")
    print("  r/R - 升高 (v_z += 0.02)")
    print("  f/F - 降低 (v_z -= 0.02)")
    print("  space/0 - 停止 (重置所有速度)")
    print("  h/H - 显示帮助")
    print("  x/X - 退出")
    print("\n当前速度:")
    print_status()
    print("\n等待输入...")

def get_key():
    """获取单个按键（非阻塞）"""
    if sys.stdin.isatty():
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())
            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = sys.stdin.read(1)
                return key
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    return None

def main():
    """主函数"""
    print("=" * 60)
    print("G1 机器人交互式速度控制")
    print("=" * 60)
    print(f"\n控制文件: {CONTROL_FILE}")
    print("提示: 此脚本会持续读取键盘输入并更新速度指令")
    print("      速度指令会写入控制文件，主脚本会读取并应用")
    print()

    # 确保控制文件存在
    os.makedirs(os.path.dirname(CONTROL_FILE) if os.path.dirname(CONTROL_FILE) else '.', exist_ok=True)
    if not os.path.exists(CONTROL_FILE):
        with open(CONTROL_FILE, 'w') as f:
            f.write(f"{current_vel[0]},{current_vel[1]},{current_vel[2]},{current_vel[3]}\n")

    print_help()

    try:
        while True:
            key = get_key()

            if key is None:
                time.sleep(0.05)
                continue

            key_lower = key.lower()

            if key_lower == 'w':
                update_velocity(delta_v_x=v_x_step)
                print_status()
            elif key_lower == 's':
                update_velocity(delta_v_x=-v_x_step)
                print_status()
            elif key_lower == 'a':
                update_velocity(delta_v_y=v_y_step)
                print_status()
            elif key_lower == 'd':
                update_velocity(delta_v_y=-v_y_step)
                print_status()
            elif key_lower == 'q':
                update_velocity(delta_v_yaw=v_yaw_step)
                print_status()
            elif key_lower == 'e':
                update_velocity(delta_v_yaw=-v_yaw_step)
                print_status()
            elif key_lower == 'r':
                update_velocity(delta_v_z=v_z_step)
                print_status()
            elif key_lower == 'f':
                update_velocity(delta_v_z=-v_z_step)
                print_status()
            elif key == ' ' or key == '0':
                current_vel[0] = 0.0
                current_vel[1] = 0.0
                current_vel[3] = 0.0
                update_velocity()
                print("\n停止: 所有速度已重置")
                print_status()
            elif key_lower == 'h':
                print_help()
            elif key_lower == 'x':
                print("\n\n退出控制程序")
                break
            elif key == '\x03':  # Ctrl+C
                print("\n\n用户中断")
                break

    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 恢复终端设置
        if sys.stdin.isatty():
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, termios.tcgetattr(sys.stdin))
        print(f"\n最终速度: v_x={current_vel[0]:.2f}, v_y={current_vel[1]:.2f}, "
              f"v_z={current_vel[2]:.2f}, v_yaw={current_vel[3]:.2f}")

if __name__ == "__main__":
    main()
