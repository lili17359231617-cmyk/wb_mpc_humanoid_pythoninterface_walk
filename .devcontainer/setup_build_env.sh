#!/usr/bin/env bash
# 新容器/新设备首次打开时：设置编译环境（ROS、子模块、环境变量）
# 用法: cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc && source .devcontainer/setup_build_env.sh
# 注意：使用 set -u 但不使用 set -e，避免某步失败时直接 exit 导致终端被关闭
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS_ROOT="$(cd "${REPO_ROOT}/../.." && pwd)"
_SETUP_ERR=0

echo "[setup_build_env] 项目目录: ${REPO_ROOT}"
echo "[setup_build_env] 工作空间: ${WS_ROOT}"

# 退出时：仅 return，不 exit，避免 source 时关闭终端
# 同时关闭 set -u，避免后续 source install/setup.bash 时因 COLCON_TRACE 等未设置变量报错
_exit_setup() {
  set +u 2>/dev/null || true
  if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    return "${1:-1}"
  else
    exit "${1:-1}"
  fi
}

# 1. ROS2 环境
if [ -f /opt/ros/jazzy/setup.bash ]; then
  source /opt/ros/jazzy/setup.bash
  echo "[setup_build_env] 已加载 ROS2 jazzy"
elif [ -f /opt/ros/humble/setup.bash ]; then
  source /opt/ros/humble/setup.bash
  echo "[setup_build_env] 已加载 ROS2 humble"
else
  echo "[setup_build_env] 未找到 /opt/ros/jazzy 或 humble，请先安装 ROS2。"
  _exit_setup 1
fi

# 2. 检查编译工具
for cmd in colcon cmake python3; do
  if ! command -v "$cmd" &>/dev/null; then
    echo "[setup_build_env] 缺少命令: $cmd，请先安装后再执行本脚本。"
    _exit_setup 1
  fi
done
echo "[setup_build_env] 编译工具检查通过"

# 3. Python 依赖（仅检查，不强制安装）
if ! python3 -c "import numpy, scipy" 2>/dev/null; then
  echo "[setup_build_env] 提示: 未检测到 numpy/scipy，请执行: pip3 install -r ${REPO_ROOT}/requirements.txt"
else
  echo "[setup_build_env] Python 依赖检查通过"
fi

# 4. 初始化 Git 子模块（失败不退出，只提示）
cd "${REPO_ROOT}"
if [ -d .git ] && [ -f .gitmodules ]; then
  if git submodule update --init --recursive; then
    echo "[setup_build_env] Git 子模块已初始化/更新"
  else
    echo "[setup_build_env] 警告: Git 子模块更新失败，可稍后手动执行: git submodule update --init --recursive"
  fi
fi

# 5. 环境变量（编译后库与 Python 路径）
INSTALL="${WS_ROOT}/install"
export LD_LIBRARY_PATH="${INSTALL}/hpipm_catkin/lib:${INSTALL}/blasfeo_catkin/lib:${LD_LIBRARY_PATH:-}"
# Python 3.12 常见路径
for pyver in python3.12 python3.11 python3; do
  PY_SITE="${INSTALL}/humanoid_wb_mpc/lib/${pyver}/site-packages"
  if [ -d "${PY_SITE}" ]; then
    export PYTHONPATH="${PY_SITE}:${PYTHONPATH:-}"
    break
  fi
done
echo "[setup_build_env] LD_LIBRARY_PATH 与 PYTHONPATH 已设置（若 install 存在）"

# 6. 并行任务数提示
if [ -z "${PARALLEL_JOBS:-}" ]; then
  echo "[setup_build_env] 建议设置: export PARALLEL_JOBS=2   # 16GB 内存用 2；32GB 用 4；64GB 用 6"
fi

# 恢复 shell 选项，避免后续 source install/setup.bash 时出现 COLCON_TRACE: unbound variable
set +u 2>/dev/null || true
echo "[setup_build_env] 完成。接下来可执行: export PARALLEL_JOBS=2 && make build-all"
