# 容器配置迁移总结

## 变更概述

已将 devcontainer 配置从使用预构建镜像模式改回构建模式，并添加了所有必要的依赖，使容器可以自动安装所有必需的库。

## 主要变更

### 1. devcontainer.json

#### ✅ 恢复构建模式
- **移除**：`"image": "g1_mpc_rl_v1"`
- **恢复**：`build` 配置，使用 `docker/Dockerfile` 构建容器

#### ✅ 简化 runArgs
- 移除了在 runArgs 中显式设置的环境变量（LD_LIBRARY_PATH, PYTHONPATH）
- 这些环境变量现在在 Dockerfile 和 postCreateCommand 中设置

#### ✅ 更新 postCreateCommand
- 添加了 LD_LIBRARY_PATH 和 PYTHONPATH 到 ~/.bashrc
- 确保每次打开终端时环境变量正确设置

### 2. Dockerfile 更新

#### ✅ 新增系统依赖
- `xvfb` - X11 虚拟帧缓冲（用于无显示器环境运行可视化程序）
- `python3-pip` - Python 包管理器
- `python3-setuptools` - Python 安装工具

#### ✅ 新增 Python 依赖安装
- 复制 `requirements.txt` 到容器
- 使用 `pip3 install` 安装所有 Python 依赖
- 使用 `--break-system-packages` 标志（适用于 Python 3.12）

#### ✅ 环境变量设置
- `LD_LIBRARY_PATH` - 包含 hpipm 和 blasfeo 库路径
- `PYTHONPATH` - 包含 humanoid_wb_mpc_py 模块路径

### 3. requirements.txt（新建）

创建了项目根目录的 `requirements.txt` 文件，包含：

#### 核心依赖（必需）
- `numpy>=1.20.0` - 数值计算库
- `scipy>=1.7.0` - 科学计算库（用于姿态计算）

#### 可选依赖（注释状态）
- `pynput>=1.7.0` - 键盘输入库（用于交互式控制）

## 安装的依赖清单

### 系统库（apt-get）
- ✅ xvfb - 虚拟显示器
- ✅ python3-pip - Python 包管理器
- ✅ python3-setuptools - Python 安装工具
- ✅ 所有原有的 dependencies.txt 中的依赖

### Python 库（pip）
- ✅ numpy>=1.20.0
- ✅ scipy>=1.7.0
- ⚠️ pynput（可选，在 requirements.txt 中注释）

## 使用方法

### 1. 构建容器

在 VS Code/Cursor 中：
- 打开命令面板（Ctrl+Shift+P / Cmd+Shift+P）
- 选择 "Dev Containers: Rebuild Container"
- 容器将自动构建并安装所有依赖

### 2. 验证安装

在容器内运行：

```bash
# 检查 Python 库
python3 -c "import numpy, scipy; print('Python dependencies OK')"

# 检查系统库
which xvfb

# 检查环境变量
echo $LD_LIBRARY_PATH
echo $PYTHONPATH
```

### 3. 运行测试脚本

```bash
cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc/humanoid_nmpc/humanoid_wb_mpc/scripts
python3 test_mpc_mrt_walk.py
```

## 可选功能

### 启用键盘控制

如果需要交互式键盘控制，编辑 `requirements.txt`：

```txt
# 取消注释以下行
pynput>=1.7.0
```

然后重新构建容器或在容器内运行：
```bash
pip3 install --break-system-packages pynput
```

## 迁移优势

1. ✅ **可移植性** - 不依赖预构建镜像，可以在任何机器上构建
2. ✅ **可重复性** - 所有依赖都在配置文件中明确声明
3. ✅ **版本控制** - Dockerfile 和 requirements.txt 可以版本控制
4. ✅ **自动化** - 容器构建时自动安装所有依赖
5. ✅ **文档化** - 依赖清单清晰可见

## 注意事项

1. **首次构建时间** - 构建容器可能需要较长时间（下载依赖、编译等）
2. **网络要求** - 需要网络连接以下载依赖
3. **Python 版本** - 当前使用 Python 3.12（在 Dockerfile 中定义）
4. **环境变量** - LD_LIBRARY_PATH 和 PYTHONPATH 在容器启动时设置，但需要编译后才会生效

## 故障排除

### 如果构建失败

1. 检查网络连接
2. 检查 Dockerfile 中的路径是否正确
3. 查看构建日志中的错误信息

### 如果 Python 模块导入失败

1. 确认已编译项目：`colcon build`
2. 检查 PYTHONPATH 是否正确设置：`echo $PYTHONPATH`
3. 确认 Python 模块已安装：`pip3 list | grep humanoid`

### 如果动态库加载失败

1. 确认已编译项目：`colcon build`
2. 检查 LD_LIBRARY_PATH：`echo $LD_LIBRARY_PATH`
3. 确认库文件存在：`ls /wb_humanoid_mpc_ws/install/hpipm_catkin/lib/`
