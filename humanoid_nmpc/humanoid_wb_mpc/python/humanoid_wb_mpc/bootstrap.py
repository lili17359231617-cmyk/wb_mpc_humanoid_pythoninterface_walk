"""
MPC Python 扩展与 MuJoCo 渲染后端引导

在导入 humanoid_wb_mpc_py 或创建 MuJoCo 环境前调用，用于：
1. 设置 MUJOCO_GL（无头/有头）
2. 预加载 libglfw / libGLEW（避免符号缺失）
3. 将 C++ 安装路径加入 sys.path

使用方式：
    import humanoid_wb_mpc.bootstrap  # 侧效应：设置环境、路径并可选导入 mpc_py
    # 然后: import humanoid_wb_mpc_py as mpc_py
"""

import os
import sys

# MuJoCo 渲染后端
if "MUJOCO_GL" not in os.environ:
    has_display = os.environ.get("DISPLAY") is not None
    is_headless = os.environ.get("HEADLESS", "0") == "1"
    if is_headless or not has_display:
        os.environ["MUJOCO_GL"] = "egl"
    else:
        os.environ["MUJOCO_GL"] = "glfw"

# C++ 扩展路径（安装目录下的 lib 或 lib/pythonX.Y/site-packages）
_ws = os.environ.get("WB_HUMANOID_MPC_WS", "/wb_humanoid_mpc_ws")
_install = os.path.join(_ws, "install", "humanoid_wb_mpc")
_py_ver = "python{}.{}".format(sys.version_info.major, sys.version_info.minor)
_candidates = [
    os.path.join(_install, "lib"),
    os.path.join(_install, "lib", _py_ver, "site-packages"),
]
for _mpc_lib in _candidates:
    if os.path.isdir(_mpc_lib) and _mpc_lib not in sys.path:
        sys.path.insert(0, _mpc_lib)
        break

# 预加载 glfw / GLEW，避免运行时符号缺失
try:
    import ctypes
    _r = ctypes.RTLD_GLOBAL
    for lib in ("libglfw.so.3", "libglfw.so"):
        try:
            ctypes.CDLL(lib, mode=_r)
            break
        except OSError:
            continue
    for lib in ("libGLEW.so.2.2", "libGLEW.so"):
        try:
            ctypes.CDLL(lib, mode=_r)
            break
        except OSError:
            continue
except Exception:
    pass


def ensure_mpc_py():
    """导入并返回 humanoid_wb_mpc_py 模块，若失败则抛出 ImportError。"""
    try:
        import humanoid_wb_mpc_py as mpc_py
        return mpc_py
    except ImportError as e:
        raise ImportError(
            f"无法导入 humanoid_wb_mpc_py: {e}\n"
            "请先编译: colcon build --packages-select humanoid_wb_mpc"
        ) from e
