# X11 环境配置：Python MuJoCo 仿真可视化

本文说明如何配置 X11，使在 **Python 控制的机器人仿真** 中能**正常看到 MuJoCo 仿真界面**（如 `test_mpc_rl.py`、`train_g1_ppo_v1.py` 等脚本的图形窗口）。

---

## 1. 使用场景概览

| 场景 | 推荐方式 |
|------|----------|
| **本机 Linux + 显示器** | 直接运行，保证 `DISPLAY` 已设置 |
| **本机 + Dev Container (Cursor/VS Code)** | 宿主机执行 `xhost`，容器已挂载 X11 socket |
| **远程 SSH 到 Linux 服务器** | SSH 开启 X11 转发（`ssh -X` / `ssh -Y`） |
| **无显示器 / 纯服务器** | 使用 Xvfb 虚拟显示 或 不开窗口（HEADLESS） |

---

## 2. 本机 + Dev Container（Cursor / VS Code）

项目 devcontainer 已做部分 X11 配置，要**在容器内看到 MuJoCo 窗口**，需满足下面条件。

### 2.1 宿主机（运行 Cursor 的机器）

- 必须是**有图形桌面的 Linux**（或 WSL2 + WSLg，见下节）。
- 在**打开 Dev Container 之前**，在宿主机终端执行一次：
  ```bash
  xhost +local:
  ```
  这样允许本机上的 Docker 容器连接你的 X Server。
  （devcontainer 的 `initializeCommand` 也会执行该命令，若仍看不到窗口，可先在宿主机手动执行。）

### 2.2 容器内环境变量（已配置）

`devcontainer.json` 中已包含：

- `DISPLAY` 从宿主机传入
- `QT_X11_NO_MITSHM=1`（避免部分 Qt/OpenGL 问题）
- 挂载 `/tmp/.X11-unix`，使容器内程序能连到宿主 X Server

一般**无需再改**。若仍无法开窗，可在容器内检查：

```bash
echo $DISPLAY
# 常见为 :0 或 :1
```

若为空，说明宿主机未传 `DISPLAY`，需在宿主机确保有图形会话并导出 `DISPLAY`（见 2.1）。

### 2.3 运行带可视化的脚本

在容器内运行（不要设 `HEADLESS=1`）。**若出现 OpenGL 报错**（如 `AttributeError: 'NoneType' object has no attribute 'glGetError'`），请先设置 `MUJOCO_GL=glfw`，让 MuJoCo 使用窗口后端而不是无头 OSMesa：

```bash
cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc
MUJOCO_GL=glfw python3 humanoid_nmpc/humanoid_wb_mpc/scripts/test_mpc_rl.py
```

或你的 PPO 训练脚本（开启 render 时）：

```bash
MUJOCO_GL=glfw python3 humanoid_nmpc/humanoid_wb_mpc/scripts/train_g1_ppo_v1.py
```

若一切正常，MuJoCo 窗口会出现在**宿主机**的桌面上。

---

## 3. 本机 Linux（不用容器）

- 确保已安装图形环境和 X11：
  ```bash
  sudo apt update
  sudo apt install -y libgl1-mesa-glx libglew2.1 x11-xserver-utils
  ```
- 设置并检查显示：
  ```bash
  echo $DISPLAY   # 通常 :0
  ```
- 直接运行 Python 脚本即可，无需 `xhost`（本机 X 默认允许本机进程）。

---

## 4. 远程 SSH 到 Linux 服务器（需要看到窗口在本机）

在**你本地电脑**上 SSH 时开启 X11 转发：

```bash
ssh -X user@server    # 或 ssh -Y user@server（信任度更高，部分 OpenGL 需要）
```

然后在服务器上：

```bash
echo $DISPLAY   # 应为 localhost:10.0 或类似
python .../test_mpc_rl.py
```

MuJoCo 窗口会通过 X11 转发显示在你**本地**显示器上。

- 若报错 `cannot open display`，在**本地**安装并启动 X Server（如 Windows 上 VcXsrv、Xming；Mac 上 XQuartz）。
- Linux 本机做 SSH 客户端时，一般已有 X，用 `ssh -X` 即可。

---

## 5. WSL2 + Cursor Dev Container

- 使用 **WSL2 + WSLg** 时，Windows 11 一般已提供显示，`DISPLAY` 由 WSL 自动设置。
- 在 WSL2 内先执行一次：
  ```bash
  xhost +local:
  ```
  再进入 Dev Container 运行脚本。
- 若仍无窗口，在 WSL2 里检查：
  ```bash
  echo $DISPLAY
  ```
  通常为 `:0`。

---

## 6. 无显示器 / 纯服务器（不要求弹窗）

两种常用方式：

### 6.1 使用 HEADLESS 模式（不打开任何窗口）

脚本里已支持通过环境变量关闭界面：

```bash
HEADLESS=1 python humanoid_nmpc/humanoid_wb_mpc/scripts/test_mpc_rl.py
```

或使用专门的无界面脚本（如带录屏的 headless 版本）：

```bash
python humanoid_nmpc/humanoid_wb_mpc/scripts/test_mpc_rl_headless.py
```

### 6.2 使用 Xvfb（虚拟显示，仅用于需要“有显示”但无物理屏幕）

在**无显示器**的机器上若仍想跑依赖 DISPLAY 的代码（不直接看窗口，但可能录屏或 offscreen 渲染），可装 Xvfb：

```bash
sudo apt install -y xvfb
```

运行时用虚拟显示：

```bash
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
python humanoid_nmpc/humanoid_wb_mpc/scripts/test_mpc_rl.py
```

窗口会开在虚拟显示上，你看不到，但程序不会因 “no display” 报错。

---

## 7. 常见错误与排查

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| `cannot open display` / `Display not set` | `DISPLAY` 未设置或未传入容器 | 宿主机执行 `xhost +local:`；SSH 用 `ssh -X`；检查 `echo $DISPLAY` |
| 窗口一闪而过或无法创建窗口 | 未允许容器连接 X | 宿主机执行 `xhost +local:` |
| `'NoneType' object has no attribute 'glGetError'`（import mujoco 即报错） | MuJoCo 默认走 OSMesa 路径且无有效上下文 | **有显示器时**在运行前设置 `MUJOCO_GL=glfw`，例如：`MUJOCO_GL=glfw python3 scripts/test_mpc_rl.py` |
| GLFW / OpenGL 其他报错 | 缺库或 GPU 驱动问题 | 安装 `libgl1-mesa-glx libglew2.1`；可尝试 `export LIBGL_ALWAYS_SOFTWARE=1` 用软件渲染 |
| 脚本自动走 HEADLESS | 检测不到 viewer | 确保有 `mujoco.viewer` 或 `mujoco_viewer`，且 `DISPLAY` 有效、未设 `HEADLESS=1` |

---

## 8. 快速检查清单（Dev Container 内要看到 MuJoCo 窗口）

1. **宿主机**：有图形桌面，并执行过 `xhost +local:`。
2. **容器内**：`echo $DISPLAY` 非空（如 `:0`）。
3. **容器内**：未设置 `HEADLESS=1`。
4. **运行**：直接运行 `test_mpc_rl.py` 或你的带 render 的脚本。

按上述配置后，Python 控制的 MuJoCo 仿真界面应能正常显示。
