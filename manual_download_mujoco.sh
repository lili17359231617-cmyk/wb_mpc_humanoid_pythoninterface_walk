#!/bin/bash
# 手动下载 MuJoCo 3.3.4 的脚本
# 使用方法：在有网络的环境中运行此脚本，然后将下载的文件放到正确位置

set -e

MUJOCO_VERSION="3.3.4"
DOWNLOAD_URL="https://github.com/google-deepmind/mujoco/releases/download/${MUJOCO_VERSION}/mujoco-${MUJOCO_VERSION}-linux-x86_64.tar.gz"
TARGET_DIR="/wb_humanoid_mpc_ws/build/mujoco_vendor/mujoco_prebuilt/src"
TARBALL_NAME="mujoco-${MUJOCO_VERSION}-linux-x86_64.tar.gz"
TARGET_FILE="${TARGET_DIR}/${TARBALL_NAME}"

echo "=========================================="
echo "手动下载 MuJoCo ${MUJOCO_VERSION}"
echo "=========================================="

# 创建目标目录
mkdir -p "${TARGET_DIR}"

# 检查文件是否已存在且有效
if [ -f "${TARGET_FILE}" ] && [ -s "${TARGET_FILE}" ]; then
    echo "✓ 文件已存在: ${TARGET_FILE}"
    echo "  文件大小: $(du -h ${TARGET_FILE} | cut -f1)"
    echo "  跳过下载"
    exit 0
fi

# 下载文件
echo "正在下载: ${DOWNLOAD_URL}"
echo "保存到: ${TARGET_FILE}"

if command -v wget &> /dev/null; then
    wget -O "${TARGET_FILE}" "${DOWNLOAD_URL}"
elif command -v curl &> /dev/null; then
    curl -L -o "${TARGET_FILE}" "${DOWNLOAD_URL}"
else
    echo "错误: 未找到 wget 或 curl"
    exit 1
fi

# 验证下载
if [ -f "${TARGET_FILE}" ] && [ -s "${TARGET_FILE}" ]; then
    echo "✓ 下载成功!"
    echo "  文件大小: $(du -h ${TARGET_FILE} | cut -f1)"
    echo "  SHA256: $(sha256sum ${TARGET_FILE} | cut -d' ' -f1)"
    echo ""
    echo "预期 SHA256: ecf1a17459a342badf2b4f32dd4677a6a0e5fd393c5143993eb3e81b8e44609b"
else
    echo "✗ 下载失败: 文件为空或不存在"
    exit 1
fi

echo ""
echo "=========================================="
echo "下载完成！现在可以重新编译："
echo "  cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc"
echo "  make build PKG=mujoco_sim_interface BUILD_WITH_NINJA=OFF"
echo "=========================================="
