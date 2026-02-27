# 🚀 快速开始 - 新容器编译指南

## ⚡ 一键编译（推荐 - 首次使用）

```bash
# 1. 运行环境设置脚本（自动检查并设置所有环境）
cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc
source .devcontainer/setup_build_env.sh

# 2. 设置并行任务数（根据内存：16GB=2, 32GB=4, 64GB=6）
export PARALLEL_JOBS=2

# 3. 编译所有包
make build-all

# 4. Source 安装目录
source /wb_humanoid_mpc_ws/install/setup.bash
```

## 📋 手动编译（如果脚本失败）

```bash
# 1. Source ROS2 环境
source /opt/ros/jazzy/setup.bash

# 2. 进入项目目录
cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc

# 3. 初始化子模块
git submodule update --init --recursive

# 4. 清理并修复权限
cd /wb_humanoid_mpc_ws
rm -rf build install log .ccache
sudo chown -R $(id -u):$(id -g) .

# 5. 设置并行任务数
export PARALLEL_JOBS=2

# 6. 编译
cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc
make build-all

# 7. Source 安装目录
source /wb_humanoid_mpc_ws/install/setup.bash
```

## 增量编译（修改代码后）

```bash
# 编译单个包
make build PKG=humanoid_wb_mpc

# Source 环境使更改生效
source /wb_humanoid_mpc_ws/install/setup.bash
```

## 验证编译

```bash
# 检查 Python 模块
python3 -c "import humanoid_wb_mpc_py; print('OK')"

# 检查库文件
ls /wb_humanoid_mpc_ws/install/hpipm_catkin/lib/
```

## 常用命令

| 命令 | 说明 |
|------|------|
| `make build-all` | 编译所有包 |
| `make build PKG=包名` | 编译单个包 |
| `make build-release` | Release 模式编译 |
| `make build-debug` | Debug 模式编译 |
| `make clean-ws` | 清理编译产物 |

## 详细文档

- 📖 **完整编译步骤**: `.devcontainer/BUILD_INSTRUCTIONS.md`
- 📋 **依赖分析**: `.devcontainer/dependency_analysis.md`
- 🔄 **迁移总结**: `.devcontainer/MIGRATION_SUMMARY.md`
