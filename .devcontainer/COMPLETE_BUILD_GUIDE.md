# 完整编译指南 - 解决编译环境问题

## 🎯 问题诊断

如果编译失败，很可能是编译环境没有正确设置。本指南提供完整的解决方案。

## 📋 完整编译流程（推荐）

### 步骤 1: 运行环境设置脚本

```bash
cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc
source .devcontainer/setup_build_env.sh
```

这个脚本会自动：
- ✅ 检查并加载 ROS2 环境
- ✅ 检查编译工具（colcon, cmake, ninja 等）
- ✅ 检查 Python 依赖（numpy, scipy）
- ✅ 初始化 Git 子模块
- ✅ 清理并修复构建目录权限
- ✅ 设置所有必要的环境变量

### 步骤 2: 设置并行任务数

```bash
# 根据系统内存设置（16GB=2, 32GB=4, 64GB=6）
export PARALLEL_JOBS=2
```

### 步骤 3: 开始编译

```bash
cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc
make build-all
```

---

## 🔍 手动检查和设置（如果脚本失败）

### 1. 检查 ROS2 环境

```bash
# 检查 ROS2 是否安装
ls /opt/ros/jazzy/setup.bash

# Source ROS2 环境
source /opt/ros/jazzy/setup.bash

# 验证
echo $ROS_DISTRO  # 应该显示: jazzy
which colcon      # 应该显示路径
```

### 2. 检查工作空间

```bash
# 确认工作空间路径
cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc
pwd  # 应该显示: /wb_humanoid_mpc_ws/src/wb_humanoid_mpc
```

### 3. 检查编译工具

```bash
# 检查所有必要的工具
colcon --version
cmake --version
ninja --version
python3 --version
pip3 --version
```

如果缺少工具，安装它们：

```bash
# ROS2 工具（应该已通过 Dockerfile 安装）
sudo apt-get update
sudo apt-get install -y python3-colcon-common-extensions

# 其他工具（应该已安装）
sudo apt-get install -y cmake ninja-build python3-pip
```

### 4. 检查 Python 依赖

```bash
# 检查依赖
python3 -c "import numpy; print('numpy OK')"
python3 -c "import scipy; print('scipy OK')"

# 如果缺少，安装它们
pip3 install --break-system-packages numpy scipy
```

### 5. 初始化 Git 子模块

```bash
cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc
git submodule update --init --recursive
```

### 6. 清理并修复权限

```bash
cd /wb_humanoid_mpc_ws

# 清理旧的构建产物
rm -rf build install log .ccache

# 修复权限
sudo chown -R $(id -u):$(id -g) /wb_humanoid_mpc_ws
```

### 7. 设置环境变量

```bash
# 设置并行任务数
export PARALLEL_JOBS=2

# 设置 ROS2 环境（如果还没 source）
source /opt/ros/jazzy/setup.bash

# 设置库路径
export LD_LIBRARY_PATH=/wb_humanoid_mpc_ws/install/hpipm_catkin/lib:/wb_humanoid_mpc_ws/install/blasfeo_catkin/lib:$LD_LIBRARY_PATH

# 设置 Python 路径
export PYTHONPATH=/wb_humanoid_mpc_ws/install/humanoid_wb_mpc/lib/python3.12/site-packages:$PYTHONPATH
```

### 8. 开始编译

```bash
cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc
make build-all
```

---

## 🐛 常见编译错误及解决方案

### 错误 1: "Permission denied" 权限错误

**症状**: `PermissionError: [Errno 13] Permission denied`

**解决**:
```bash
cd /wb_humanoid_mpc_ws
sudo chown -R $(id -u):$(id -g) .
rm -rf build install log .ccache
```

### 错误 2: "Could not find a package"

**症状**: `Could not find a package configuration file provided by "xxx"`

**解决**:
```bash
# 确保已 source ROS2 环境
source /opt/ros/jazzy/setup.bash

# 如果编译了部分包，source 安装目录
source /wb_humanoid_mpc_ws/install/setup.bash
```

### 错误 3: "colcon: command not found"

**症状**: `colcon: command not found`

**解决**:
```bash
# Source ROS2 环境
source /opt/ros/jazzy/setup.bash

# 或者安装 colcon
sudo apt-get install -y python3-colcon-common-extensions
```

### 错误 4: "ninja: error: loading 'build.ninja'"

**症状**: `ninja: error: loading 'build.ninja': No such file or directory`

**解决**:
```bash
# 清理构建目录
cd /wb_humanoid_mpc_ws
rm -rf build install log .ccache

# 重新编译
cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc
make build-all
```

### 错误 5: Python 模块导入失败

**症状**: `ImportError: No module named 'xxx'`

**解决**:
```bash
# 安装 Python 依赖
pip3 install --break-system-packages numpy scipy

# 如果编译后仍无法导入，检查 PYTHONPATH
export PYTHONPATH=/wb_humanoid_mpc_ws/install/humanoid_wb_mpc/lib/python3.12/site-packages:$PYTHONPATH
```

---

## ✅ 编译成功后的验证

### 1. 检查编译产物

```bash
ls /wb_humanoid_mpc_ws/install/
# 应该看到多个包的目录
```

### 2. Source 安装目录

```bash
source /wb_humanoid_mpc_ws/install/setup.bash
```

### 3. 验证 Python 模块

```bash
python3 -c "import humanoid_wb_mpc_py; print('Python module OK')"
```

### 4. 验证库文件

```bash
ls /wb_humanoid_mpc_ws/install/hpipm_catkin/lib/
ls /wb_humanoid_mpc_ws/install/blasfeo_catkin/lib/
```

---

## 🔄 一键编译脚本

创建一个完整的编译脚本 `build.sh`:

```bash
#!/bin/bash
set -e

echo "=========================================="
echo "开始编译 Whole Body MPC"
echo "=========================================="

# 1. 设置环境
cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc
source .devcontainer/setup_build_env.sh

# 2. 设置并行任务数
export PARALLEL_JOBS=${PARALLEL_JOBS:-2}

# 3. 编译
echo ""
echo "开始编译..."
make build-all

# 4. Source 安装目录
echo ""
echo "Source 安装目录..."
source /wb_humanoid_mpc_ws/install/setup.bash

# 5. 验证
echo ""
echo "验证编译结果..."
python3 -c "import humanoid_wb_mpc_py; print('✓ Python module OK')" || echo "⚠ Python module not found (may need to compile humanoid_wb_mpc)"

echo ""
echo "=========================================="
echo "编译完成！"
echo "=========================================="
```

使用方式：
```bash
chmod +x build.sh
./build.sh
```

---

## 📝 重要提示

1. **每次打开新终端**都要 source ROS2 环境：
   ```bash
   source /opt/ros/jazzy/setup.bash
   ```

2. **编译后**要 source 安装目录：
   ```bash
   source /wb_humanoid_mpc_ws/install/setup.bash
   ```

3. **添加到 ~/.bashrc**（可选，自动设置）：
   ```bash
   echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
   echo "source /wb_humanoid_mpc_ws/install/setup.bash" >> ~/.bashrc
   ```

4. **首次编译**可能需要 30-60 分钟，请耐心等待

5. **内存不足**时减少并行任务数：
   ```bash
   export PARALLEL_JOBS=1
   ```

---

## 📚 相关文档

- **快速开始**: `.devcontainer/QUICK_START.md`
- **详细编译步骤**: `.devcontainer/BUILD_INSTRUCTIONS.md`
- **权限问题**: `.devcontainer/FIX_PERMISSIONS.md`
