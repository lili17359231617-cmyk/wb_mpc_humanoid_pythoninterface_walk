#!/bin/bash
# 修复 catkin 权限问题的脚本

set -e

echo "=========================================="
echo "修复 catkin 构建权限问题"
echo "=========================================="

WORKSPACE="/wb_humanoid_mpc_ws"

# 1. 清理有问题的 catkin 构建目录
if [ -d "$WORKSPACE/build/catkin" ]; then
    echo "清理 catkin 构建目录..."
    rm -rf "$WORKSPACE/build/catkin"
    echo "✓ 已清理"
fi

# 2. 确保所有脚本文件有执行权限
echo ""
echo "修复所有脚本文件权限..."
find "$WORKSPACE/build" -type f \( -name "*.sh" -o -name "env.sh" -o -name "setup.sh" \) 2>/dev/null | while read file; do
    if [ -f "$file" ]; then
        chmod +x "$file" 2>/dev/null || true
    fi
done
echo "✓ 权限修复完成"

# 3. 确保目录有正确的权限
echo ""
echo "修复目录权限..."
find "$WORKSPACE/build" -type d -exec chmod 755 {} \; 2>/dev/null || true
echo "✓ 目录权限修复完成"

# 4. 提供手动修复命令
echo ""
echo "提示: 如果编译时仍遇到权限问题，可以运行："
echo "  find $WORKSPACE/build -name 'env.sh' -exec chmod +x {} \\;"

echo ""
echo "=========================================="
echo "修复完成！"
echo "=========================================="
echo ""
echo "现在可以重新编译了："
echo "  cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc"
echo "  make build PKG=humanoid_mpc_msgs BUILD_WITH_NINJA=OFF"
echo ""
echo "如果仍然遇到权限问题，可以在另一个终端运行监控脚本："
echo "  $WORKSPACE/fix_permissions_monitor.sh"
echo ""
