# G1 机器人稳定行走控制脚本使用指南

## 目录
1. [宿主机直接运行](#宿主机直接运行)
2. [Docker容器中运行](#docker容器中运行)
3. [键盘控制说明](#键盘控制说明)
4. [常见问题](#常见问题)

---

## 宿主机直接运行

### 前置要求

1. **安装Python依赖**
```bash
pip install numpy scipy pynput
# 或者
pip install numpy scipy keyboard
```

2. **确保C++扩展模块已编译**
```bash
cd /wb_humanoid_mpc_ws
colcon build --packages-select humanoid_wb_mpc
source install/setup.bash
```

### 运行脚本

**方式1：有图形界面（推荐）**
```bash
cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc/humanoid_nmpc/humanoid_wb_mpc/scripts
python3 test_mpc_mrt_walk.py
```

**方式2：无图形界面（使用虚拟显示器）**
```bash
# 启动虚拟显示器
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99

# 运行脚本
cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc/humanoid_nmpc/humanoid_wb_mpc/scripts
python3 test_mpc_mrt_walk.py
```

---

## Docker容器中运行

### 方法1：使用Xvfb虚拟显示器 + 键盘输入（推荐）

#### 步骤1：在Docker容器中启动虚拟显示器

```bash
# 进入Docker容器
docker exec -it <container_name> bash

# 在容器内启动Xvfb（如果未安装，先安装：apt-get update && apt-get install -y xvfb）
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99

# 安装Python键盘库（如果未安装）
pip install pynput
# 或者
pip install keyboard
```

#### 步骤2：运行脚本

```bash
cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc/humanoid_nmpc/humanoid_wb_mpc/scripts
python3 test_mpc_mrt_walk.py
```

#### 步骤3：从宿主机发送键盘输入（可选）

由于Docker容器内可能无法直接接收键盘输入，可以使用以下方法：

**方法A：使用xdotool模拟键盘输入**
```bash
# 在宿主机上安装xdotool
sudo apt-get install xdotool

# 发送键盘事件到Docker容器（需要配置X11转发）
# 注意：这需要X11转发配置，比较复杂
```

**方法B：使用网络socket通信（需要修改脚本）**
- 可以修改脚本添加socket服务器，从宿主机发送控制命令

**方法C：使用环境变量或文件传递命令（简单方法）**
- 修改脚本支持从环境变量或文件读取速度指令
- 在宿主机上修改文件，脚本定期读取

### 方法2：使用Docker的X11转发（如果容器支持GUI）

**步骤1：在宿主机终端允许X11转发**

```bash
xhost +local:docker
# 或者更宽松（仅限本地）：
xhost +local:
```

**步骤2：查看本机Docker镜像名**

```bash
docker images
# 找到你的镜像名和标签，例如：g1_mpc_rl_v1 或 wb-humanoid-mpc:dev
```

**步骤3：在宿主机运行容器（把下面的镜像名换成你上一步看到的）**

```bash
# 使用项目 devcontainer 的镜像名（一般为 g1_mpc_rl_v1）
docker run -it \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /wb_humanoid_mpc_ws:/wb_humanoid_mpc_ws \
  g1_mpc_rl_v1 bash
```

若工作空间不在宿主机的 `/wb_humanoid_mpc_ws`，请改成实际路径，例如：

```bash
docker run -it \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$(pwd)/../..":/wb_humanoid_mpc_ws \
  g1_mpc_rl_v1 bash
```

**步骤4：在容器内运行脚本**

```bash
cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc/humanoid_nmpc/humanoid_wb_mpc/scripts
python3 test_mpc_mrt_walk.py
```

**说明：** 若你的镜像名不是 `g1_mpc_rl_v1`，把上面命令里最后一行的 `g1_mpc_rl_v1` 换成 `docker images` 里显示的镜像名（如 `wb-humanoid-mpc`）和标签（如 `:dev`）。

### 方法3：修改脚本使用环境变量控制（最简单）

如果键盘输入在Docker中不可用，可以修改脚本支持环境变量：

```bash
# 在宿主机终端设置速度指令
export VEL_X=0.2
export VEL_Y=0.0
export VEL_Z=0.85
export VEL_YAW=0.0

# 在Docker容器中运行
docker exec -it <container_name> bash -c "cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc/humanoid_nmpc/humanoid_wb_mpc/scripts && python3 test_mpc_mrt_walk.py"
```

---

## 键盘控制说明

### 控制键映射

| 按键 | 功能 | 说明 |
|------|------|------|
| `W` | 前进 | 增加 v_x（向前速度） |
| `S` | 后退 | 减少 v_x（向后速度） |
| `A` | 左移 | 增加 v_y（向左速度） |
| `D` | 右移 | 减少 v_y（向右速度） |
| `Q` | 左转 | 增加 v_yaw（逆时针旋转） |
| `E` | 右转 | 减少 v_yaw（顺时针旋转） |
| `R` | 升高 | 增加 v_z（基座高度） |
| `F` | 降低 | 减少 v_z（基座高度） |
| `Space` | 停止 | 将所有速度设为0 |

### 速度限制

- **v_x** (前后速度): -0.5 ~ 0.5 m/s
- **v_y** (左右速度): -0.3 ~ 0.3 m/s
- **v_z** (基座高度): 0.7 ~ 1.0 m
- **v_yaw** (旋转速度): -0.5 ~ 0.5 rad/s

### 速度增量

每次按键的速度变化量：
- v_x_step = 0.1 m/s
- v_y_step = 0.1 m/s
- v_z_step = 0.02 m
- v_yaw_step = 0.2 rad/s

---

## 常见问题

### Q1: 键盘输入无响应

**问题确认：**
如果运行 `test_keyboard_simple.py` 后10秒内未检测到按键，说明键盘监听在容器中无法工作。

**可能原因：**
- Docker容器无法访问键盘设备（最常见）
- 键盘库未安装
- 虚拟显示器未正确配置

**解决方法：**

**方案1：使用交互式控制脚本（推荐）**
```bash
# 终端1：运行主脚本（启用文件控制）
export USE_FILE_CONTROL=1
python3 test_mpc_mrt_walk.py

# 终端2：运行交互式控制
python3 interactive_control.py
# 然后按 w/s/a/d/q/e/r/f 控制，按 x 退出
```

**方案2：使用控制脚本**
```bash
# 终端1：运行主脚本
export USE_FILE_CONTROL=1
python3 test_mpc_mrt_walk.py

# 终端2：发送命令
./control_velocity.sh forward
./control_velocity.sh stop
```

**方案3：直接写入文件**
```bash
echo "0.2,0.0,0.85,0.0" >> /tmp/velocity_control.txt
```

### Q2: 在Docker中无法使用键盘

**解决方法：**
- 使用环境变量方式（见方法3）
- 修改脚本添加socket通信
- 使用文件传递控制命令

### Q3: 机器人不移动

**检查项：**
1. 查看终端输出的速度指令值（Cmd行）
2. 确认MPC策略已就绪（启动时会显示"就绪！"）
3. 检查目标轨迹是否正常更新（每0.5秒打印一次状态）

### Q4: MuJoCo 仿真界面没有打开

**可能原因和解决方法：**

1. **HEADLESS 模式已启用**
```bash
# 检查
echo $HEADLESS

# 如果显示 1，则禁用它
export HEADLESS=0
# 或
unset HEADLESS
```

2. **DISPLAY 环境变量未设置**
```bash
# 检查
echo $DISPLAY

# 如果为空，设置它
export DISPLAY=:0  # 使用主显示器
# 或使用虚拟显示器
export DISPLAY=:99
```

3. **X11 权限问题**
```bash
# 在宿主机上执行（不是容器内）
xhost +local:
```

4. **使用诊断脚本**
```bash
# 运行诊断脚本
cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc/humanoid_nmpc/humanoid_wb_mpc/scripts
./check_display.sh
```

5. **在 Docker 容器中**
```bash
# 确保容器启动时传递了 DISPLAY
docker run -it \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  ...

# 在容器内检查
echo $DISPLAY
echo $HEADLESS
```

**快速修复（在运行脚本前执行）：**
```bash
export HEADLESS=0
export DISPLAY=:0
# 然后运行脚本
python3 test_mpc_mrt_walk.py
```

### Q5: 虚拟显示器启动失败

```bash
# 检查Xvfb是否安装
which Xvfb

# 如果未安装，安装它
apt-get update
apt-get install -y xvfb

# 检查DISPLAY环境变量
echo $DISPLAY
```

### Q5: 性能问题

如果控制循环运行缓慢：
- 降低控制频率（修改 `control_frequency`）
- 降低轨迹更新频率（修改 `trajectory_update_frequency`）
- 检查系统资源使用情况

---

## 快速开始示例

### 宿主机快速测试

```bash
# 1. 安装依赖
pip install numpy scipy pynput

# 2. 编译（如果需要）
cd /wb_humanoid_mpc_ws
colcon build --packages-select humanoid_wb_mpc
source install/setup.bash

# 3. 运行
cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc/humanoid_nmpc/humanoid_wb_mpc/scripts
python3 test_mpc_mrt_walk.py

# 4. 使用键盘控制
# 按 W 键开始前进
# 按 Space 键停止
```

### Docker容器快速测试

```bash
# 1. 进入容器
docker exec -it <container_name> bash

# 2. 启动虚拟显示器
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99

# 3. 安装键盘库
pip install pynput

# 4. 运行脚本
cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc/humanoid_nmpc/humanoid_wb_mpc/scripts
python3 test_mpc_mrt_walk.py

# 注意：在Docker中键盘输入可能不可用，机器人将保持站立状态
```

---

## 调试技巧

1. **查看实时状态**
   - 脚本每0.5秒打印一次状态信息
   - 关注 `Cmd` 行，查看当前速度指令

2. **检查MPC状态**
   - 启动时会显示MPC策略就绪状态
   - 如果未就绪，检查MPC初始化是否成功

3. **查看错误信息**
   - 脚本包含详细的错误处理和打印
   - 检查终端输出的错误信息

4. **性能监控**
   - 脚本会报告控制循环的实际运行时间
   - 如果运行缓慢，会显示警告信息
