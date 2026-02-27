# 容器依赖对比分析报告

## 一、devcontainer.json 配置变化

### 1.1 主要变化
- **从构建模式改为使用预构建镜像**：
  - 原来：使用 `build` 配置，从 `docker/Dockerfile` 构建
  - 现在：直接使用预构建镜像 `g1_mpc_rl_v1`

### 1.2 新增环境变量配置
- **LD_LIBRARY_PATH**：解决 libhpipm.so 和 libblasfeo.so 动态库加载问题
  ```json
  "/wb_humanoid_mpc_ws/install/hpipm_catkin/lib:/wb_humanoid_mpc_ws/install/blasfeo_catkin/lib"
  ```
  ✅ **必要** - 这些库是 OCS2 MPC 求解器必需的

- **PYTHONPATH**：确保能直接导入 humanoid_wb_mpc_py 模块
  ```json
  "/wb_humanoid_mpc_ws/install/humanoid_wb_mpc/lib/python3.12/site-packages"
  ```
  ✅ **必要** - Python 绑定模块需要此路径

### 1.3 其他变化
- 注释掉了 `--rm` 参数（容器退出时不自动删除）
  ⚠️ **可选** - 取决于是否需要保留容器状态

---

## 二、原本 Dockerfile 中的依赖（dependencies.txt）

### 2.1 系统库（C/C++）
- `clang-format` - 代码格式化工具
- `make`, `build-essential` - 编译工具链
- `libeigen3-dev` - 线性代数库 ✅ **必要**
- `libglpk-dev` - 线性规划求解器 ✅ **必要** (OCS2 需要)
- `libboost-all-dev`, `libboost-filesystem-dev`, `libboost-log-dev` - Boost 库 ✅ **必要**
- `libglfw3-dev` - OpenGL 窗口库 ✅ **必要** (MuJoCo 渲染需要)
- `liburdfdom-dev` - URDF 解析库 ✅ **必要**
- `libabsl-dev` - Google Abseil 库 ✅ **必要**
- `libxinerama-dev`, `libxcursor-dev`, `libxi-dev` - X11 输入库 ✅ **必要** (GLFW 需要)
- `ninja-build` - 构建系统
- `ccache` - 编译缓存

### 2.2 Python 库（系统包）
- `python3-pygame` - Pygame 库
- `python3-tk` - Tkinter GUI 库
- `black` - Python 代码格式化工具

### 2.3 ROS2 包（Jazzy 发行版）
- `ros-jazzy-ament-cmake-clang-format`
- `ros-jazzy-joint-state-publisher-gui`
- `ros-jazzy-xacro` ✅ **必要**
- `ros-jazzy-mcap-vendor`
- `ros-jazzy-interactive-markers`
- `ros-jazzy-pinocchio` ✅ **必要**
- `ros-jazzy-rviz2`
- `ros-jazzy-rosidl-typesupport-fastrtps-c`

---

## 三、为混合架构行走功能可能新增的依赖

### 3.1 Python 库（pip install）

#### ✅ **必要的库**
1. **numpy** - 数值计算库
   - 用途：机器人状态、轨迹计算
   - 来源：`test_mpc_mrt_walk.py` 中使用
   - 建议：添加到 Dockerfile 或 requirements.txt

2. **scipy** - 科学计算库
   - 用途：`scipy.spatial.transform.Rotation` 用于姿态计算
   - 来源：`test_mpc_mrt_walk.py` 中使用
   - 建议：添加到 Dockerfile 或 requirements.txt

3. **pynput** - 键盘输入库
   - 用途：交互式控制机器人行走（W/S/A/D/Q/E/R/F 键）
   - 来源：`test_mpc_mrt_walk.py` 中可选使用
   - 建议：
     - 如果不需要交互式键盘控制，可以**不安装**
     - 如果需要，建议使用 `pynput`（更现代，跨平台）
     - 可以通过文件控制替代（`USE_FILE_CONTROL=1`）

### 3.2 系统库（apt-get install）

#### ✅ **必要的库（如果使用虚拟显示器）**
1. **xvfb** - X11 虚拟帧缓冲
   - 用途：在无显示器的环境中运行需要 DISPLAY 的程序
   - 来源：`README_walk.md` 中提到
   - 建议：如果容器内需要运行 MuJoCo 可视化，建议安装

#### ⚠️ **可选的库**
2. **xdotool** - X11 自动化工具
   - 用途：模拟键盘输入（用于 Docker 容器）
   - 来源：`README_walk.md` 中提到
   - 建议：**可能冗余** - 如果使用文件控制或环境变量控制，不需要此库

3. **libgl1-mesa-glx** - Mesa OpenGL 库
   - 用途：MuJoCo 渲染支持
   - 来源：`docs/X11_MuJoCo_Display_Setup.md`
   - 建议：如果使用 MuJoCo 可视化，**可能必要**（但 Dockerfile 中已有 `mesa-utils`，可能已包含）

4. **libglew2.1** - OpenGL 扩展库
   - 用途：OpenGL 功能扩展
   - 来源：`docs/X11_MuJoCo_Display_Setup.md`
   - 建议：**可能冗余** - CMakeLists.txt 中已要求 `GLEW`，可能已通过其他方式安装

5. **x11-xserver-utils** - X11 服务器工具
   - 用途：X11 相关工具（如 `xhost`）
   - 来源：`docs/X11_MuJoCo_Display_Setup.md`
   - 建议：**可能冗余** - 通常 X11 环境已包含

---

## 四、依赖分类总结

### 4.1 ✅ **必要依赖**（必须安装）

#### Python 库
- `numpy` - 数值计算
- `scipy` - 科学计算（特别是 `scipy.spatial.transform`）

#### 系统库
- `xvfb` - 虚拟显示器（如果容器内需要运行可视化）

#### 环境变量配置
- `LD_LIBRARY_PATH` - hpipm/blasfeo 库路径
- `PYTHONPATH` - Python 模块路径

### 4.2 ⚠️ **可选依赖**（根据使用场景）

#### Python 库
- `pynput` 或 `keyboard` - 仅当需要交互式键盘控制时安装
  - 替代方案：使用文件控制（`USE_FILE_CONTROL=1`）或环境变量控制

#### 系统库
- `xdotool` - 可能冗余，文件控制更简单
- `libgl1-mesa-glx` - 可能已包含在 `mesa-utils` 中
- `libglew2.1` - 可能已通过 CMake 依赖安装
- `x11-xserver-utils` - 可能已包含在基础 X11 环境中

### 4.3 ❌ **可能冗余的依赖**

1. **xdotool** - 如果使用文件控制或环境变量控制，不需要
2. **libglew2.1** - 如果 CMakeLists.txt 已正确处理 GLEW，可能冗余
3. **x11-xserver-utils** - 基础 X11 环境通常已包含

---

## 五、建议的依赖管理方案

### 5.1 创建 requirements.txt（Python 依赖）

建议在项目根目录或 `humanoid_wb_mpc/scripts/` 目录创建 `requirements.txt`：

```txt
# 核心依赖（必要）
numpy>=1.20.0
scipy>=1.7.0

# 可选依赖（交互式控制）
# pynput>=1.7.0  # 取消注释以启用键盘控制
```

### 5.2 更新 Dockerfile（系统依赖）

如果使用预构建镜像 `g1_mpc_rl_v1`，建议在镜像构建时添加：

```dockerfile
# 安装虚拟显示器（如果需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
RUN pip3 install --no-cache-dir \
    numpy>=1.20.0 \
    scipy>=1.7.0
    # pynput>=1.7.0  # 可选：取消注释以启用键盘控制
```

### 5.3 在 devcontainer.json 中添加 postCreateCommand

可以在 `postCreateCommand` 中添加依赖安装：

```json
"postCreateCommand": "git config --global user.name \"$GIT_USER_NAME\" && git config --global user.email \"$GIT_USER_EMAIL\" && git config --global --add safe.directory /wb_humanoid_mpc_ws/src/wb_humanoid_mpc && curl -o .git-completion.bash https://raw.githubusercontent.com/git/git/master/contrib/completion/git-completion.bash && echo 'source /opt/ros/jazzy/setup.bash' >> ~/.bashrc && echo 'source ./.git-completion.bash' >> ~/.bashrc && pip3 install --user numpy scipy"
```

---

## 六、检查清单

### 6.1 检查已安装的 Python 库
```bash
pip3 list | grep -E "(numpy|scipy|pynput|keyboard)"
```

### 6.2 检查已安装的系统库
```bash
dpkg -l | grep -E "(xvfb|xdotool|libgl1-mesa|libglew|x11-xserver)"
```

### 6.3 检查环境变量
```bash
echo $LD_LIBRARY_PATH
echo $PYTHONPATH
```

---

## 七、总结

### 7.1 必须安装的依赖
- ✅ `numpy` (Python)
- ✅ `scipy` (Python)
- ✅ `xvfb` (系统库，如果使用虚拟显示器)
- ✅ 环境变量：`LD_LIBRARY_PATH`, `PYTHONPATH`

### 7.2 可选安装的依赖
- ⚠️ `pynput` 或 `keyboard` (Python，仅当需要键盘控制)
- ⚠️ `xdotool` (系统库，可能冗余)

### 7.3 可能冗余的依赖
- ❌ `libglew2.1` (可能已通过其他方式安装)
- ❌ `x11-xserver-utils` (基础环境可能已包含)
- ❌ `libgl1-mesa-glx` (可能已包含在 `mesa-utils` 中)

### 7.4 建议
1. **最小化安装**：只安装 `numpy`, `scipy`, `xvfb` 和必要的环境变量
2. **按需安装**：`pynput` 等交互式控制库根据实际需求安装
3. **文档化**：在 README 或 requirements.txt 中明确记录依赖
