#!/usr/bin/env bash
# 安装本项目所有 Python 依赖（根 + 子包），便于迁移到新设备后一键安装
# 用法: cd /path/to/wb_humanoid_mpc && .devcontainer/install_python_deps.sh
# 容器内若用系统 Python: PIP_EXTRA="--break-system-packages" .devcontainer/install_python_deps.sh
# 或者 pip install -r requirements.txt --break-system-packages
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PIP_EXTRA="${PIP_EXTRA:---user}"

echo "[install_python_deps] 安装根目录 requirements.txt ..."
pip3 install ${PIP_EXTRA} -r requirements.txt

echo "[install_python_deps] 安装 humanoid_common_mpc_pyutils 依赖 ..."
pip3 install ${PIP_EXTRA} -r humanoid_nmpc/humanoid_common_mpc_pyutils/requirements.txt

if [ "${INSTALL_REMOTE_CONTROL:-0}" = "1" ]; then
  echo "[install_python_deps] 安装 remote_control (GUI) 依赖 ..."
  pip3 install ${PIP_EXTRA} -r humanoid_nmpc/remote_control/requirements.txt
else
  echo "[install_python_deps] 跳过 remote_control（需时设 INSTALL_REMOTE_CONTROL=1）"
fi

echo "[install_python_deps] 完成"
