# 新容器打开后重新编译项目的具体步骤

## 📋 前置检查清单

在开始编译之前，请确认以下事项：

### 1. 检查容器环境

```bash
# 检查 ROS2 环境
echo $ROS_DISTRO
# 应该显示: jazzy

# 检查工作空间路径
pwd
# 应该在: /wb_humanoid_mpc_ws/src/wb_humanoid_mpc

# 检查环境变量
echo $LD_LIBRARY_PATH
echo $PYTHONPATH
```

### 2. 检查依赖是否已安装

```bash
# 检查 Python 依赖
python3 -c "import numpy, scipy; print('Python dependencies OK')"

# 检查系统工具
which colcon
which cmake
which ninja-build
```

### 3. 初始化 Git 子模块（如果尚未初始化）

```bash
cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc
git submodule update --init --recursive
```

---

## 🚀 编译步骤

### 方法一：使用 Makefile（推荐）

#### 步骤 1: 设置并行编译任务数（根据系统内存）

```bash
# 根据系统内存选择合适的并行任务数
# 16GB RAM -> 2 个任务（默认）
# 32GB RAM -> 4 个任务
# 64GB RAM -> 6 个任务

export PARALLEL_JOBS=2  # 根据你的系统内存调整
```

#### 步骤 2: 编译所有包

```bash
cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc
make build-all
```

**预计时间**: 首次编译可能需要 30-60 分钟（取决于系统性能）

#### 步骤 3: Source 安装目录

编译完成后，需要 source 安装目录以使用编译好的包：

```bash
cd /wb_humanoid_mpc_ws
source install/setup.bash
```

**注意**: 这个命令需要在每次打开新终端时执行，或者将其添加到 `~/.bashrc`：

```bash
echo "source /wb_humanoid_mpc_ws/install/setup.bash" >> ~/.bashrc
```

---

### 方法二：编译特定包（增量编译）

如果只需要编译特定包或修改了部分代码：

#### 编译单个包

```bash
cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc
make build PKG=humanoid_wb_mpc
```

#### 编译多个相关包

```bash
# 编译 humanoid_wb_mpc 及其所有依赖
make build PKG=humanoid_wb_mpc
```

#### 常用包的编译命令

```bash
# 编译核心 MPC 包
make build PKG=humanoid_wb_mpc

# 编译机器人模型包
make build PKG=g1_description

# 编译 G1 机器人配置包
make build PKG=g1_wb_mpc

# 编译仿真接口
make build PKG=mujoco_sim_interface
```

---

### 方法三：使用 colcon 直接编译（高级用法）

如果需要更多控制：

```bash
cd /wb_humanoid_mpc_ws

# Source ROS2 环境
source /opt/ros/jazzy/setup.bash

# 编译所有包
colcon build \
    --parallel-workers 2 \
    --symlink-install \
    --cmake-args \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Source 安装目录
source install/setup.bash
```

---

## 🔧 编译选项说明

### 编译类型选择

```bash
# Release 模式（优化性能，默认）
make build-release

# Debug 模式（包含调试信息）
make build-debug

# RelWithDebInfo 模式（优化 + 调试信息）
make build-relwithdebinfo
```

### 禁用测试编译（加快编译速度）

```bash
# Release 模式 + 禁用测试
make build-release BUILD_TESTING=OFF
```

### 使用 Make 而不是 Ninja（如果遇到问题）

```bash
make build-all BUILD_WITH_NINJA=OFF
```

---

## ✅ 验证编译结果

### 1. 检查编译是否成功

```bash
# 检查安装目录
ls -la /wb_humanoid_mpc_ws/install/

# 应该看到以下目录：
# - humanoid_wb_mpc/
# - humanoid_common_mpc/
# - mujoco_sim_interface/
# - 等等...
```

### 2. 检查 Python 模块

```bash
# Source 环境
source /wb_humanoid_mpc_ws/install/setup.bash

# 检查 Python 模块是否可以导入
python3 -c "import humanoid_wb_mpc_py; print('Python module OK')"
```

### 3. 检查动态库

```bash
# 检查 hpipm 和 blasfeo 库
ls /wb_humanoid_mpc_ws/install/hpipm_catkin/lib/
ls /wb_humanoid_mpc_ws/install/blasfeo_catkin/lib/

# 应该看到 .so 文件
```

### 4. 运行测试（可选）

```bash
# 运行 MPC 相关测试
make run-mpc-tests

# 运行 OCS2 相关测试
make run-ocs2-tests
```

---

## 🐛 常见问题排查

### 问题 0: 权限错误 - Permission denied

**症状**: `PermissionError: [Errno 13] Permission denied: '/wb_humanoid_mpc_ws/build/catkin/devel/env.sh'`

**解决方法**:

**快速修复**:
```bash
# 运行修复脚本
cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc
./.devcontainer/fix_build_permissions.sh

# 然后重新编译
make build-all
```

**手动修复**:
```bash
# 1. 清理构建目录
cd /wb_humanoid_mpc_ws
rm -rf build install log .ccache

# 2. 修复权限
sudo chown -R $(id -u):$(id -g) /wb_humanoid_mpc_ws

# 3. 重新编译
cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc
make build-all
```

**详细说明**: 查看 `.devcontainer/FIX_PERMISSIONS.md`

---

### 问题 1: 编译失败 - 内存不足

**症状**: 编译过程中系统卡死或报错 "out of memory"

**解决方法**:
```bash
# 减少并行任务数
export PARALLEL_JOBS=1
make build-all
```

### 问题 2: Python 模块导入失败

**症状**: `ImportError: No module named 'humanoid_wb_mpc_py'`

**解决方法**:
```bash
# 1. 确认已编译
make build PKG=humanoid_wb_mpc

# 2. Source 环境
source /wb_humanoid_mpc_ws/install/setup.bash

# 3. 检查 PYTHONPATH
echo $PYTHONPATH
# 应该包含: /wb_humanoid_mpc_ws/install/humanoid_wb_mpc/lib/python3.12/site-packages

# 4. 如果 PYTHONPATH 未设置，手动设置
export PYTHONPATH=/wb_humanoid_mpc_ws/install/humanoid_wb_mpc/lib/python3.12/site-packages:$PYTHONPATH
```

### 问题 3: 动态库加载失败

**症状**: `error while loading shared libraries: libhpipm.so`

**解决方法**:
```bash
# 1. 确认库已编译
ls /wb_humanoid_mpc_ws/install/hpipm_catkin/lib/

# 2. 检查 LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH
# 应该包含: /wb_humanoid_mpc_ws/install/hpipm_catkin/lib:/wb_humanoid_mpc_ws/install/blasfeo_catkin/lib

# 3. 如果未设置，手动设置
export LD_LIBRARY_PATH=/wb_humanoid_mpc_ws/install/hpipm_catkin/lib:/wb_humanoid_mpc_ws/install/blasfeo_catkin/lib:$LD_LIBRARY_PATH
```

### 问题 4: 编译时找不到依赖包

**症状**: `Could not find a package configuration file provided by "xxx"`

**解决方法**:
```bash
# 1. 先编译依赖包
make build PKG=humanoid_common_mpc

# 2. Source 环境
source /wb_humanoid_mpc_ws/install/setup.bash

# 3. 再编译目标包
make build PKG=humanoid_wb_mpc
```

### 问题 5: Git 子模块未初始化

**症状**: 编译时找不到某些文件或目录

**解决方法**:
```bash
cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc
git submodule update --init --recursive
```

---

## 📝 完整编译流程示例

以下是一个完整的首次编译流程：

```bash
# 1. 进入项目目录
cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc

# 2. 初始化 Git 子模块
git submodule update --init --recursive

# 3. 设置并行任务数（根据系统内存）
export PARALLEL_JOBS=2

# 4. 编译所有包
make build-all

# 5. Source 安装目录
cd /wb_humanoid_mpc_ws
source install/setup.bash

# 6. 验证编译结果
python3 -c "import humanoid_wb_mpc_py; print('✓ Python module OK')"
ls /wb_humanoid_mpc_ws/install/hpipm_catkin/lib/ && echo "✓ Libraries OK"

# 7. 测试运行（可选）
cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc/humanoid_nmpc/humanoid_wb_mpc/scripts
python3 test_mpc_mrt_walk.py
```

---

## 🔄 增量编译（修改代码后）

如果修改了代码，只需要重新编译相关包：

```bash
# 1. Source 环境（如果新终端）
source /wb_humanoid_mpc_ws/install/setup.bash

# 2. 编译修改的包
cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc
make build PKG=humanoid_wb_mpc

# 3. 重新 source（使更改生效）
source /wb_humanoid_mpc_ws/install/setup.bash
```

---

## 🧹 清理编译产物

如果需要完全重新编译：

```bash
# 清理所有编译产物
cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc
make clean-ws

# 然后重新编译
make build-all
```

---

## 📚 相关资源

- **项目 README**: `/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/README.md`
- **Makefile**: `/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/Makefile`
- **依赖分析**: `.devcontainer/dependency_analysis.md`
- **迁移总结**: `.devcontainer/MIGRATION_SUMMARY.md`

---

## 💡 提示

1. **首次编译**: 建议使用 `PARALLEL_JOBS=2` 以避免内存问题
2. **日常开发**: 使用 `make build PKG=包名` 进行增量编译
3. **环境变量**: 每次打开新终端都要 source `install/setup.bash`
4. **自动补全**: 可以将 source 命令添加到 `~/.bashrc` 中
5. **编译时间**: 首次编译可能需要 30-60 分钟，后续增量编译通常只需几分钟
