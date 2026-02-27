#!/bin/bash
# 编译 humanoid_wb_mpc Python 绑定模块的脚本

set -e

echo "=========================================="
echo "编译 humanoid_wb_mpc Python 绑定模块"
echo "=========================================="

# 进入工作空间
cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc

# 检查是否已 source ROS2 环境
if [ -z "$ROS_DISTRO" ]; then
    echo "⚠ 警告: ROS_DISTRO 未设置，尝试 source ROS2 环境..."
    if [ -f "/opt/ros/jazzy/setup.bash" ]; then
        source /opt/ros/jazzy/setup.bash
        echo "✓ 已 source ROS2 Jazzy 环境"
    elif [ -f "/opt/ros/humble/setup.bash" ]; then
        source /opt/ros/humble/setup.bash
        echo "✓ 已 source ROS2 Humble 环境"
    else
        echo "✗ 错误: 未找到 ROS2 安装，请先安装 ROS2"
        exit 1
    fi
fi

# Source 已安装的包（如果存在）
if [ -f "/wb_humanoid_mpc_ws/install/setup.bash" ]; then
    source /wb_humanoid_mpc_ws/install/setup.bash
    echo "✓ 已 source 安装目录"
fi

echo ""
echo "步骤 0: 修复权限问题..."
echo "----------------------------------------"
# 修复 catkin env.sh 文件的权限问题
find /wb_humanoid_mpc_ws/build -name "env.sh" -type f -exec chmod +x {} \; 2>/dev/null || true

# 如果 catkin 构建有问题，清理并重新开始
if [ -d "/wb_humanoid_mpc_ws/build/catkin" ] && [ ! -x "/wb_humanoid_mpc_ws/build/catkin/devel/env.sh" ] 2>/dev/null; then
    echo "⚠ 检测到 catkin 权限问题，清理构建目录..."
    rm -rf /wb_humanoid_mpc_ws/build/catkin
    echo "✓ 已清理 catkin 构建目录"
fi

# 确保所有脚本文件都有执行权限
find /wb_humanoid_mpc_ws/build -type f \( -name "*.sh" -o -name "env.sh" \) -exec chmod +x {} \; 2>/dev/null || true
echo "✓ 权限检查完成"

echo ""
echo "步骤 1: 检查依赖包..."
echo "----------------------------------------"

# 检查必要的依赖包是否已编译
REQUIRED_PACKAGES=(
    "robot_core"
    "robot_model"
    "mujoco_vendor"
    "mujoco_sim_interface"
    "humanoid_common_mpc"
    "humanoid_mpc_msgs"
)

MISSING_PACKAGES=()
for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if [ ! -d "/wb_humanoid_mpc_ws/install/$pkg" ]; then
        MISSING_PACKAGES+=("$pkg")
    else
        echo "  ✓ $pkg 已安装"
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo ""
    echo "⚠ 以下依赖包未编译，需要先编译："
    for pkg in "${MISSING_PACKAGES[@]}"; do
        echo "  - $pkg"
    done
    echo ""
    echo "正在编译依赖包..."

    for pkg in "${MISSING_PACKAGES[@]}"; do
        echo "  编译 $pkg..."
        # 在后台运行权限修复（如果编译时间较长）
        (
            while true; do
                find /wb_humanoid_mpc_ws/build -name "env.sh" -type f -exec chmod +x {} \; 2>/dev/null || true
                sleep 0.5
            done
        ) &
        PERM_FIX_PID=$!

        # 编译包
        make build PKG="$pkg" BUILD_WITH_NINJA=OFF || {
            kill $PERM_FIX_PID 2>/dev/null || true
            echo "✗ 编译 $pkg 失败"
            exit 1
        }

        # 停止权限修复进程
        kill $PERM_FIX_PID 2>/dev/null || true

        # 最后修复一次权限
        find /wb_humanoid_mpc_ws/build -name "env.sh" -type f -exec chmod +x {} \; 2>/dev/null || true

        source /wb_humanoid_mpc_ws/install/setup.bash
    done
fi

echo ""
echo "步骤 2: 编译 humanoid_wb_mpc 包..."
echo "----------------------------------------"

# 在后台运行权限修复
(
    while true; do
        find /wb_humanoid_mpc_ws/build -name "env.sh" -type f -exec chmod +x {} \; 2>/dev/null || true
        sleep 0.5
    done
) &
PERM_FIX_PID=$!

# 编译 humanoid_wb_mpc 包
make build PKG=humanoid_wb_mpc BUILD_WITH_NINJA=OFF || {
    kill $PERM_FIX_PID 2>/dev/null || true
    echo "✗ 编译失败"
    exit 1
}

# 停止权限修复进程
kill $PERM_FIX_PID 2>/dev/null || true

# 最后修复一次权限
find /wb_humanoid_mpc_ws/build -name "env.sh" -type f -exec chmod +x {} \; 2>/dev/null || true

echo ""
echo "步骤 3: Source 安装目录..."
echo "----------------------------------------"
source /wb_humanoid_mpc_ws/install/setup.bash

echo ""
echo "步骤 4: 验证 Python 模块..."
echo "----------------------------------------"

# 检查 Python 模块是否存在
PYTHON_MODULE_PATH="/wb_humanoid_mpc_ws/install/humanoid_wb_mpc/lib"
if [ -f "$PYTHON_MODULE_PATH/humanoid_wb_mpc_py.so" ] || \
   [ -f "$PYTHON_MODULE_PATH/humanoid_wb_mpc_py.cpython-*.so" ]; then
    echo "✓ Python 模块文件存在"
    ls -lh "$PYTHON_MODULE_PATH"/humanoid_wb_mpc_py* 2>/dev/null || true
else
    # 检查其他可能的位置
    ALT_PATH=$(find /wb_humanoid_mpc_ws/install/humanoid_wb_mpc -name "humanoid_wb_mpc_py*.so" 2>/dev/null | head -1)
    if [ -n "$ALT_PATH" ]; then
        echo "✓ Python 模块文件存在于: $ALT_PATH"
    else
        echo "⚠ 警告: 未找到 Python 模块文件"
        echo "  检查路径: $PYTHON_MODULE_PATH"
        ls -la "$PYTHON_MODULE_PATH" 2>/dev/null || echo "  目录不存在"
    fi
fi

# 尝试导入模块
echo ""
echo "尝试导入 Python 模块..."
python3 -c "import humanoid_wb_mpc_py; print('✓ Python 模块导入成功')" 2>&1 || {
    echo "⚠ Python 模块导入失败，可能需要设置 PYTHONPATH"
    echo ""
    echo "手动设置 PYTHONPATH:"
    echo "  export PYTHONPATH=$PYTHON_MODULE_PATH:\$PYTHONPATH"
    echo ""
    echo "或者 source 安装目录:"
    echo "  source /wb_humanoid_mpc_ws/install/setup.bash"
}

echo ""
echo "=========================================="
echo "编译完成！"
echo "=========================================="
echo ""
echo "使用前请确保 source 安装目录:"
echo "  source /wb_humanoid_mpc_ws/install/setup.bash"
echo ""
