#!/usr/bin/env bash
# 迁移验证脚本：在新设备上运行，检验环境与编译结果是否满足运行条件。
# 用法: cd /path/to/wb_humanoid_mpc && ./.devcontainer/verify_migration.sh
# 可选: INSTALL_DIR=/path/to/ws/install ./.devcontainer/verify_migration.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 安装目录：优先环境变量，否则默认为容器内路径或 repo 上两级/install
INSTALL_DIR="${INSTALL_DIR:-$(cd "${REPO_ROOT}/../.." 2>/dev/null && pwd)/install}"
FAILED=0

echo "=============================================="
echo "  迁移验证：wb_humanoid_mpc"
echo "  项目目录: ${REPO_ROOT}"
echo "  安装目录: ${INSTALL_DIR}"
echo "=============================================="

# 1. 系统 / ROS 环境
check_ros() {
  if [ -f /opt/ros/jazzy/setup.bash ] || [ -f /opt/ros/humble/setup.bash ]; then
    echo "[通过] ROS2 已安装"
  else
    echo "[失败] 未找到 ROS2 (jazzy/humble)"
    FAILED=1
  fi
}

# 2. 编译工具
check_tools() {
  for cmd in colcon cmake python3 make; do
    if command -v "$cmd" &>/dev/null; then
      echo "[通过] $cmd: $(command -v "$cmd")"
    else
      echo "[失败] 未找到命令: $cmd"
      FAILED=1
    fi
  done
}

# 3. Python 依赖（不 source install 也可检查 pip 包）
check_python_deps() {
  if python3 -c "import numpy, scipy, gymnasium, mujoco, stable_baselines3" 2>/dev/null; then
    echo "[通过] Python 依赖 (numpy, scipy, gymnasium, mujoco, stable_baselines3)"
  else
    echo "[失败] 缺少 Python 依赖，请执行: pip3 install -r ${REPO_ROOT}/requirements.txt"
    python3 -c "import numpy, scipy, gymnasium, mujoco, stable_baselines3" 2>/dev/null || true
    FAILED=1
  fi
}

# 4. 安装目录与 C++ 库（编译后才存在）
check_install() {
  if [ ! -d "${INSTALL_DIR}" ]; then
    echo "[跳过] 安装目录不存在 (尚未编译): ${INSTALL_DIR}"
    echo "       编译后请重新运行本脚本验证"
    return
  fi
  if [ -f "${INSTALL_DIR}/setup.bash" ]; then
    echo "[通过] 安装目录存在且含 setup.bash"
  else
    echo "[失败] 安装目录无 setup.bash: ${INSTALL_DIR}"
    FAILED=1
  fi
  for lib in hpipm_catkin blasfeo_catkin; do
    if [ -d "${INSTALL_DIR}/${lib}/lib" ] && ls "${INSTALL_DIR}/${lib}/lib"/*.so* &>/dev/null; then
      echo "[通过] ${lib} 库已安装"
    else
      echo "[失败] 未找到 ${lib} 库: ${INSTALL_DIR}/${lib}/lib"
      FAILED=1
    fi
  done
}

# 5. humanoid_wb_mpc_py（需 source install 后才有 PYTHONPATH）
check_mpc_py() {
  if [ ! -f "${INSTALL_DIR}/setup.bash" ]; then
    echo "[跳过] humanoid_wb_mpc_py（需先编译并 source install）"
    return
  fi
  # 临时 source 后检查（set +u 避免 setup.bash 中未设置变量报错）
  if (set +u; source "${INSTALL_DIR}/setup.bash" 2>/dev/null && python3 -c "import humanoid_wb_mpc_py; print('OK')" 2>/dev/null); then
    echo "[通过] humanoid_wb_mpc_py 可导入"
  else
    echo "[失败] 无法导入 humanoid_wb_mpc_py（请先 make build-all 并 source install/setup.bash）"
    FAILED=1
  fi
}

# 6. Git 子模块
check_submodules() {
  cd "${REPO_ROOT}"
  if [ -f .gitmodules ]; then
    uninit=$(git submodule status --recursive 2>/dev/null | grep -c '^-' || true)
    if [ "${uninit:-0}" -eq 0 ]; then
      echo "[通过] Git 子模块已初始化"
    else
      echo "[失败] 有 ${uninit} 个子模块未初始化，请执行: git submodule update --init --recursive"
      FAILED=1
    fi
  else
    echo "[通过] 无子模块或非 git 仓库，跳过"
  fi
}

check_ros
check_tools
check_python_deps
check_submodules
check_install
check_mpc_py

echo "=============================================="
if [ $FAILED -eq 0 ]; then
  echo "  结果: 全部通过，可认为迁移/环境就绪"
  echo "=============================================="
  exit 0
else
  echo "  结果: 存在失败项，请按上述提示修复后重试"
  echo "=============================================="
  exit 1
fi
