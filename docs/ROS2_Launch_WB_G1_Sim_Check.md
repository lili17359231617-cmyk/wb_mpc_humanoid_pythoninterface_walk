# ROS2 launch-wb-g1-sim 运行检查说明

## 当前结论

- **launch 无法直接运行的原因**：当前 workspace 的 `install/` 中**没有** `g1_wb_mpc` 包，因此执行 `make launch-wb-g1-sim` 会报错：
  ```text
  Package 'g1_wb_mpc' not found
  ```
- **MPC 逻辑未被破坏**：你已用 `test_mpc_rl.py` 跑通 MuJoCo + MPC（SQP 收敛、仿真步进正常），说明 `WBMpcInterface`、`MujocoSimInterface`、`WBMpcMrtJointController` 等核心逻辑是正常的。ROS2 launch 只是另一条入口，用的是同一套 C++ 节点 `humanoid_wb_mpc_sim`（即 `WBMpcRobotSim.cpp`）。

## launch 与 C++ 节点对应关系

| 项目 | 说明 |
|------|------|
| 命令 | `make launch-wb-g1-sim` → `ros2 launch g1_wb_mpc mujoco_sim.launch.py` |
| Launch 包 | `robot_models/unitree_g1/g1_wb_mpc/launch/mujoco_sim.launch.py` |
| 配置类 | `humanoid_common_mpc_ros2/mpc_launch_config.py` 中的 `MPCLaunchConfig` |
| 可执行文件 | `humanoid_wb_mpc_sim`（由 `humanoid_wb_mpc_ros2` 包提供，源码 `WBMpcRobotSim.cpp`） |
| 节点参数顺序 | `robot_name`, `config_name`(task.info), `target_command_file`(reference.info), `description_name`(urdf), `target_gait_file`(gait.info), `xml_path`(mjx) |

`WBMpcRobotSim.cpp` 中 `main()` 期望的 6 个参数与上述顺序一致，**launch 与 C++ 逻辑是对齐的**。

## 如何让 launch-wb-g1-sim 真正跑起来

1. **在 workspace 根目录下完整构建并安装 `g1_wb_mpc`**（会连带构建 `humanoid_wb_mpc_ros2`、MuJoCo 等）：
   ```bash
   cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc
   make build PKG=g1_wb_mpc BUILD_WITH_NINJA=OFF
   ```
   - 若之前用 Unix Makefiles 建过 build，请保持 `BUILD_WITH_NINJA=OFF`，否则会报 CMake 生成器不一致。
   - 若 `mujoco_vendor` 从 GitHub 下载失败，需解决网络/代理或使用本地 MuJoCo，再重跑上述构建。

2. **构建成功后启动仿真**（必须在**仓库目录**下执行 `make`，因为 Makefile 在此）：
   ```bash
   cd /wb_humanoid_mpc_ws/src/wb_humanoid_mpc
   make launch-wb-g1-sim
   ```
   需要具备图形与 MuJoCo 显示（如 `MUJOCO_GL=glfw` 或 EGL 等）。

3. **必须同时安装 `g1_description`**：launch 在初始化时会调用 `get_package_share_directory("g1_description")` 获取 URDF/XML 路径，若未安装会报错：
   ```text
   PackageNotFoundError: "package 'g1_description' not found"
   ```
   请单独构建并安装：
   ```bash
   make build PKG=g1_description BUILD_WITH_NINJA=OFF
   ```
   （已在 `g1_wb_mpc/package.xml` 中加入对 `g1_description` 的依赖，今后用 `make build PKG=g1_wb_mpc` 会一并构建 `g1_description`。）

4. **可选：只验证 launch 文件语法**（不跑节点）：
   ```bash
   cd /wb_humanoid_mpc_ws
   source /opt/ros/jazzy/setup.bash
   source install/setup.bash
   ros2 launch g1_wb_mpc mujoco_sim.launch.py --show-arguments
   ```
   只有在 `g1_wb_mpc` **和** `g1_description` 都已安装时该命令才会成功。

## 小结

- 当前 **ROS2 launch-wb-g1-sim 不能运行** 是因为 **未安装 `g1_wb_mpc`**，与“之前改框架是否打乱 MPC 逻辑”无关。
- **MPC 运行逻辑**已由 `test_mpc_rl.py` 验证正常；完成上述构建并安装 `g1_wb_mpc` 后，再执行 `make launch-wb-g1-sim` 即可在 ROS2 下复验同一套 MPC 仿真。
