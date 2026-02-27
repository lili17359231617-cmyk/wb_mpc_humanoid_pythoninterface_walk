/**
 * @file python_binding.cpp
 * @brief G1 人形机器人全身控制循环的 Python 绑定
 *
 * 本模块提供以下功能的 Python 绑定:
 * - WBMpcInterface: 全身控制的 MPC 求解器接口
 * - WBMpcMrtJointController: 基于 MRT 的关节控制器
 * - MpcWeightAdjustmentModule: 用于 RL 集成的权重调整模块
 * - RobotState/RobotJointAction: 机器人状态和动作表示
 * - MujocoSimInterface: Mujoco 仿真接口
 * - WBMpcTargetTrajectoriesCalculator: 将用户指令（位置/速度）转换为 MPC 目标轨迹
 * - TargetTrajectories: MPC 目标轨迹数据结构
 * - SwitchedModelReferenceManager: 切换模型参考管理器（步态相位、接触状态）
 * - SwingTrajectoryPlanner: 摆动腿规划器（摆动脚 z 位置/速度/加速度约束）
 *
 * @author 人形机器人 MPC 团队
 * @date 2025
 */

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

// MPC 和控制器头文件
#include <ocs2_core/reference/TargetTrajectories.h>
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"
#include "humanoid_wb_mpc/WBMpcInterface.h"
#include "humanoid_wb_mpc/command/WBMpcTargetTrajectoriesCalculator.h"
#include "humanoid_wb_mpc/mrt/WBMpcMrtJointController.h"
#include "humanoid_wb_mpc/synchronized_module/MpcWeightAdjustmentModule.h"

// Robot model头文件
#include "robot_model/RobotDescription.h"
#include "robot_model/RobotHWInterfaceBase.h"
#include "robot_model/RobotJointAction.h"
#include "robot_model/RobotState.h"

// MujocoSimInterface - 仅在mujoco_sim_interface包存在时启用
#include "mujoco_sim_interface/MujocoSimInterface.h"

namespace py = pybind11;
using namespace ocs2::humanoid;
using namespace robot::model;

// =============================================================================
// 用于 STL/optional 绑定的辅助结构体
// =============================================================================

// std::vector/std::optional 特殊处理的透明类型声明
PYBIND11_MAKE_OPAQUE(std::vector<JointAction>);
PYBIND11_MAKE_OPAQUE(std::vector<bool>);
PYBIND11_MAKE_OPAQUE(std::optional<JointAction>);

PYBIND11_MODULE(humanoid_wb_mpc_py, m) {
  m.doc() = R"(
    G1 人形机器人全身 MPC 控制的 Python 绑定

    本模块提供以下核心接口:
    1. WBMpcInterface - 全身控制的 MPC 求解器接口
    2. WBMpcMrtJointController - 基于 MRT 的关节控制器
    3. MpcWeightAdjustmentModule - 用于 RL 集成的权重调整模块
    4. RobotState - 机器人状态表示 (关节, 基座位姿, 速度)
    5. RobotJointAction - 机器人关节指令 (位置, 速度, 力矩)
    6. MujocoSimInterface - Mujoco 物理仿真接口
    7. WBMpcTargetTrajectoriesCalculator - 将用户指令转换为 MPC 目标轨迹
    8. TargetTrajectories - MPC 目标轨迹数据结构

    使用示例:
        # 基本 MPC 控制
        interface = WBMpcInterface(task_file, urdf_file, ref_file)
        interface.setup_mpc()
        controller = WBMpcMrtJointController(interface, frequency=100.0)
        controller.start_mpc_thread(initial_state)
        actions = controller.compute_joint_control_action(time, state)

        # 通过键盘输入生成目标轨迹
        calculator = WBMpcTargetTrajectoriesCalculator(ref_file, interface)
        # 速度指令: [v_x, v_y, v_z, v_yaw]
        target_traj = calculator.commanded_velocity_to_target_trajectories(
            commanded_velocities=[0.5, 0.0, 0.0, 0.0],
            init_time=0.0,
            init_state=current_state
        )
  )";

  // =============================================================================
  // 1. 机器人模型类型绑定
  // =============================================================================

  // 1.1 RobotDescription 机器人描述类
  py::class_<RobotDescription>(m, "RobotDescription")
      .def(py::init<const std::string&>(), py::arg("urdf_path"))
      .def("get_num_joints", &RobotDescription::getNumJoints)
      // 使用 lambda 显式绑定无参数版本的 getJointIndices
      .def(
          "get_joint_indices", [](RobotDescription& self) -> const std::vector<robot::joint_index_t>& { return self.getJointIndices(); },
          "返回关节索引列表")
      .def("get_joint_names", &RobotDescription::getJointNames, "返回关节名称列表")
      .def("get_joint_index", &RobotDescription::getJointIndex, py::arg("joint_name"), "根据关节名称获取索引")
      .def("get_joint_name", &RobotDescription::getJointName, py::arg("joint_index"), "根据关节索引获取名称")
      .def("get_urdf_path", &RobotDescription::getURDFPath)
      .def("__repr__",
           [](const RobotDescription& self) { return "<RobotDescription with " + std::to_string(self.getNumJoints()) + " joints>"; });

  // 1.2 JointState 关节状态类
  py::class_<JointState>(m, "JointState")
      .def(py::init<>())
      .def_readwrite("position", &JointState::position)
      .def_readwrite("velocity", &JointState::velocity)
      .def_readwrite("measured_effort", &JointState::measuredEffort)
      .def("__repr__", [](const JointState& self) {
        return "<JointState pos=" + std::to_string(self.position) + ", vel=" + std::to_string(self.velocity) + ">";
      });

  // 1.3 RobotState 机器人状态类
  py::class_<RobotState>(m, "RobotState")
      .def(py::init<const RobotDescription&, size_t>(), py::arg("robot_description"), py::arg("contact_size") = 2)

      // 基座姿态获取器
      .def("get_root_position", &RobotState::getRootPositionInWorldFrame, "获取基座位置 [x, y, z] (世界坐标系)")
      .def(
          "get_root_rotation_quat",
          [](const RobotState& self) -> Eigen::Vector4d {
            robot::quaternion_t quat = self.getRootRotationLocalToWorldFrame();
            // 返回 [w, x, y, z] 格式（MuJoCo 格式）
            Eigen::Vector4d result;
            result[0] = quat.w();
            result[1] = quat.x();
            result[2] = quat.y();
            result[3] = quat.z();
            return result;
          },
          "获取基座姿态四元数 [w, x, y, z] (MuJoCo 格式)")

      // 基座线速度获取器
      .def("get_root_linear_velocity", &RobotState::getRootLinearVelocityInLocalFrame, "获取基座线速度 (本地坐标系)")
      .def("get_root_angular_velocity", &RobotState::getRootAngularVelocityInLocalFrame, "获取基座角速度 (本地坐标系)")

      // 基座姿态设置器
      .def("set_root_position", &RobotState::setRootPositionInWorldFrame, py::arg("position"), "设置基座位置 [x, y, z] (世界坐标系)")
      .def(
          "set_root_rotation_quat",
          [](RobotState& self, const Eigen::Ref<const Eigen::Vector4d>& quat_array) {
            // 从 numpy 数组创建 Eigen::Quaternion
            // 支持两种格式: [w, x, y, z] (MuJoCo) 或 [x, y, z, w] (scipy)
            // Eigen::Quaternion 构造函数: (w, x, y, z)
            // 假设输入是 [w, x, y, z] 格式（MuJoCo 格式）
            robot::quaternion_t quat(quat_array[0], quat_array[1], quat_array[2], quat_array[3]);
            self.setRootRotationLocalToWorldFrame(quat);
          },
          py::arg("quaternion"), "设置基座姿态四元数 (numpy 数组 [w, x, y, z])")

      // 基座线速度设置器
      .def("set_root_linear_velocity", &RobotState::setRootLinearVelocityInLocalFrame, py::arg("velocity"), "设置基座线速度 (本地坐标系)")
      .def("set_root_angular_velocity", &RobotState::setRootAngularVelocityInLocalFrame, py::arg("velocity"), "设置基座角速度 (本地坐标系)")

      // 关节访问器
      .def("set_joint_position", &RobotState::setJointPosition, py::arg("joint_id"), py::arg("position"))
      .def("get_joint_position", &RobotState::getJointPosition, py::arg("joint_id"))
      .def("set_joint_velocity", &RobotState::setJointVelocity, py::arg("joint_id"), py::arg("velocity"))
      .def("get_joint_velocity", &RobotState::getJointVelocity, py::arg("joint_id"))

      // 接触标志
      .def("get_contact_flag", &RobotState::getContactFlag, py::arg("index"), "获取末端执行器的接触标志")
      .def("set_contact_flag", &RobotState::setContactFlag, py::arg("index"), py::arg("contact_flag"), "设置末端执行器的接触标志")
      .def("get_contact_flags", &RobotState::getContactFlags, "获取所有接触标志 (布尔值列表)")

      // 时间
      .def("get_time", &RobotState::getTime)
      .def("set_time", &RobotState::setTime, py::arg("time"))

      // 零位配置
      .def("set_configuration_to_zero", &RobotState::setConfigurationToZero)

      .def("__repr__", [](const RobotState& self) { return "<RobotState t=" + std::to_string(self.getTime()) + ">"; });

  // 1.4 JointAction 关节动作类
  py::class_<JointAction>(m, "JointAction")
      .def(py::init<>())
      .def_readwrite("q_des", &JointAction::q_des, "期望关节位置")
      .def_readwrite("qd_des", &JointAction::qd_des, "期望关节速度")
      .def_readwrite("kp", &JointAction::kp, "位置增益 (刚度)")
      .def_readwrite("kd", &JointAction::kd, "速度增益 (阻尼)")
      .def_readwrite("feed_forward_effort", &JointAction::feed_forward_effort, "前馈力矩")
      .def("get_total_feedback_torque", &JointAction::getTotalFeedbackTorque, py::arg("current_q"), py::arg("current_qd"),
           "计算反馈力矩: kp*(q_des - q) + kd*(qd_des - qd) + ff_effort")
      .def("__repr__", [](const JointAction& self) {
        return "<JointAction q_des=" + std::to_string(self.q_des) + ", kp=" + std::to_string(self.kp) + ">";
      });

  // 1.5 RobotJointAction (基于 JointIdMap 的容器类，用于存储所有关节的动作)
  py::class_<RobotJointAction, std::unique_ptr<RobotJointAction>>(m, "RobotJointAction")
      .def(py::init<const RobotDescription&>(), py::arg("robot_description"))
      .def(
          "at", [](RobotJointAction& self, size_t joint_id) -> JointAction& { return self.at(joint_id).value(); }, py::arg("joint_id"),
          "获取指定关节 ID 的动作引用")
      .def(
          "__getitem__", [](RobotJointAction& self, size_t joint_id) -> JointAction& { return self.at(joint_id).value(); },
          py::arg("joint_id"), "使用索引访问关节动作")
      .def("__len__", [](RobotJointAction& self) { return self.size(); }, "返回关节数量");

  // =============================================================================
  // 2. 模型设置绑定
  // =============================================================================

  py::class_<ModelSettings::FootConstraintConfig>(m, "FootConstraintConfig")
      .def(py::init<>())
      .def_readwrite("position_error_gain_z", &ModelSettings::FootConstraintConfig::positionErrorGain_z)
      .def_readwrite("orientation_error_gain", &ModelSettings::FootConstraintConfig::orientationErrorGain)
      .def_readwrite("linear_velocity_error_gain_z", &ModelSettings::FootConstraintConfig::linearVelocityErrorGain_z)
      .def_readwrite("linear_velocity_error_gain_xy", &ModelSettings::FootConstraintConfig::linearVelocityErrorGain_xy)
      .def_readwrite("angular_velocity_error_gain", &ModelSettings::FootConstraintConfig::angularVelocityErrorGain)
      .def_readwrite("linear_acceleration_error_gain_z", &ModelSettings::FootConstraintConfig::linearAccelerationErrorGain_z)
      .def_readwrite("linear_acceleration_error_gain_xy", &ModelSettings::FootConstraintConfig::linearAccelerationErrorGain_xy)
      .def_readwrite("angular_acceleration_error_gain", &ModelSettings::FootConstraintConfig::angularAccelerationErrorGain);

  py::class_<ModelSettings>(m, "ModelSettings")
      .def_readonly("robot_name", &ModelSettings::robotName)
      .def_readonly("mpc_joint_dim", &ModelSettings::mpc_joint_dim)
      .def_readonly("full_joint_dim", &ModelSettings::full_joint_dim)
      .def_readonly("mpc_model_joint_names", &ModelSettings::mpcModelJointNames)
      .def_readonly("contact_names", &ModelSettings::contactNames)
      .def_readonly("foot_constraint_config", &ModelSettings::footConstraintConfig);

  // =============================================================================
  // 2.5 摆动腿规划器与参考管理器（供 Python 查询摆动相约束与接触状态）
  // =============================================================================

  py::class_<SwingTrajectoryPlanner, std::shared_ptr<SwingTrajectoryPlanner>>(
      m, "SwingTrajectoryPlanner",
      R"(摆动腿规划器：根据步态相位生成摆动脚的高度/速度/加速度约束。
通常通过 interface.get_swing_trajectory_planner() 或 ref_manager.get_swing_trajectory_planner() 获取。)")
      .def("get_z_position_constraint", &SwingTrajectoryPlanner::getZpositionConstraint, py::arg("leg"), py::arg("time"),
           "给定腿索引与时间，返回该脚在 z 方向的期望位置约束 (m)")
      .def("get_z_velocity_constraint", &SwingTrajectoryPlanner::getZvelocityConstraint, py::arg("leg"), py::arg("time"),
           "给定腿索引与时间，返回该脚在 z 方向的期望速度约束 (m/s)")
      .def("get_z_acceleration_constraint", &SwingTrajectoryPlanner::getZaccelerationConstraint, py::arg("leg"), py::arg("time"),
           "给定腿索引与时间，返回该脚在 z 方向的期望加速度约束 (m/s^2)")
      .def("get_impact_proximity_factor", &SwingTrajectoryPlanner::getImpactProximityFactor, py::arg("leg"), py::arg("time"),
           "给定腿索引与时间，返回着地邻近因子 (用于代价/约束缩放)")
      .def("__repr__", [](const SwingTrajectoryPlanner&) { return "<SwingTrajectoryPlanner>"; });

  py::class_<SwitchedModelReferenceManager, std::shared_ptr<SwitchedModelReferenceManager>>(
      m, "SwitchedModelReferenceManager",
      R"(切换模型参考管理器：管理步态相位、接触状态与摆动腿规划器。
通过 interface.get_switched_model_reference_manager_ptr() 获取。)")
      .def("get_swing_trajectory_planner", &SwitchedModelReferenceManager::getSwingTrajectoryPlanner,
           "返回摆动腿规划器 (SwingTrajectoryPlanner) 的 shared_ptr")
      .def(
          "get_contact_flags",
          [](const SwitchedModelReferenceManager& self, ocs2::scalar_t time) {
            const auto flags = self.getContactFlags(time);
            return std::vector<bool>(flags.begin(), flags.end());
          },
          py::arg("time"), "返回当前时刻各脚的接触标志 [左脚, 右脚]，True 表示着地")
      .def("is_in_stance_phase", &SwitchedModelReferenceManager::isInStancePhase, py::arg("time"), "当前时刻是否处于双足支撑相")
      .def("is_in_contact", &SwitchedModelReferenceManager::isInContact, py::arg("time"), py::arg("contact_index"),
           "指定脚在当前时刻是否着地 (contact_index: 0=左, 1=右)")
      .def("get_phase_variable", &SwitchedModelReferenceManager::getPhaseVariable, py::arg("time"), "返回当前步态相位变量")
      .def("get_gait_schedule", &SwitchedModelReferenceManager::getGaitSchedule, "返回步态调度器 (GaitSchedule) 的 shared_ptr")
      .def("__repr__", [](const SwitchedModelReferenceManager&) { return "<SwitchedModelReferenceManager>"; });

  // =============================================================================
  // 3. WBMpcInterface 绑定
  // =============================================================================

  py::class_<WBMpcInterface>(m, "WBMpcInterface")
      .def(py::init<const std::string&, const std::string&, const std::string&, bool>(), py::arg("task_file"), py::arg("urdf_file"),
           py::arg("reference_file"), py::arg("setup_ocp") = true,
           R"(
           创建全身控制的 MPC 接口。

           参数:
               task_file: MPC 任务配置文件的路径
               urdf_file: 机器人 URDF 文件的路径
               reference_file: 参考配置文件的路径
               setup_ocp: 是否立即设置最优控制问题
           )")

      // MPC 设置
      .def("setup_mpc", &WBMpcInterface::setupMpc, "初始化并设置 MPC 求解器")
      .def("reset", &WBMpcInterface::reset, "重置 MPC 和 MRT 内部状态")

      // MPC 执行
      .def("run_mpc", &WBMpcInterface::runMpc, py::arg("observation"), py::arg("time"),
           R"(
           运行 MPC 求解器一个时间步。

           参数:
               observation: 完整的机器人状态向量
               time: 当前仿真时间

           返回:
               最优控制输入向量 (关节加速度/力矩)
           )")

      // 状态/输入维度
      .def("get_state_dim", &WBMpcInterface::getStateDim, "获取 MPC 状态向量的维度")
      .def("get_input_dim", &WBMpcInterface::getInputDim, "获取 MPC 输入向量的维度")

      // 内部组件获取器
      .def("get_model_settings", &WBMpcInterface::modelSettings, py::return_value_policy::reference, "获取 MPC 模型设置")
      .def("get_pinocchio_interface", &WBMpcInterface::getPinocchioInterface, py::return_value_policy::reference,
           "获取 Pinocchio 接口用于运动学/动力学计算")
      .def("get_mpc_ptr", &WBMpcInterface::getMpcPtr, py::return_value_policy::reference, "获取原始 MPC 指针 (高级用法)")

      // 参考轨迹管理
      .def("set_target_state", &WBMpcInterface::setTargetState, py::arg("target_state"), "设置 MPC 轨迹的目标状态")
      .def("set_target_trajectories", &WBMpcInterface::setTargetTrajectories, py::arg("target_trajectories"),
           "设置 MPC 的完整目标轨迹（包含时间序列、状态序列和输入序列）")

      // 用于 RL 的权重调整
      .def("update_mpc_weights", &WBMpcInterface::updateMpcWeights, py::arg("new_weights"),
           R"(
           根据 RL 策略输出更新 MPC 权重。

           参数:
               new_weights: 新的权重值的 Eigen 向量
           )")
      .def("get_weight_adjustment_module", &WBMpcInterface::getWeightAdjustmentModule, py::return_value_policy::reference,
           "获取用于 RL 集成的权重调整模块")

      // 初始状态
      .def("get_initial_state", &WBMpcInterface::getInitialState, "获取 MPC 的初始状态向量")

      // 参考与摆动腿规划
      .def("get_switched_model_reference_manager_ptr", &WBMpcInterface::getSwitchedModelReferenceManagerPtr,
           R"(返回切换模型参考管理器 (SwitchedModelReferenceManager)，可查询接触状态、步态相位与摆动腿规划器。)")
      .def(
          "get_swing_trajectory_planner",
          [](WBMpcInterface& self) { return self.getSwitchedModelReferenceManagerPtr()->getSwingTrajectoryPlanner(); },
          R"(返回摆动腿规划器 (SwingTrajectoryPlanner)，可查询给定时刻各脚的高度/速度/加速度约束。)")

      .def("__repr__", [](const WBMpcInterface& self) {
        return "<WBMpcInterface state_dim=" + std::to_string(self.getStateDim()) + ", input_dim=" + std::to_string(self.getInputDim()) +
               ">";
      });

  // =============================================================================
  // 4. WBMpcMrtJointController 绑定
  // =============================================================================

  py::class_<WBMpcMrtJointController>(m, "WBMpcMrtJointController")
      .def(py::init([](WBMpcInterface& interface, double frequency) {
             // 使用 URDF 路径创建 RobotDescription
             auto robotDescription = std::make_unique<RobotDescription>(interface.getURDFFile());
             return new WBMpcMrtJointController(*robotDescription, interface.modelSettings(), *interface.getMpcPtr(),
                                                interface.getPinocchioInterface(), frequency,
                                                nullptr  // 纯 Python 模式下不启用 RViz 可视化
             );
           }),
           py::arg("interface"), py::arg("frequency") = 100.0,
           R"(
           创建基于 MRT 的关节控制器。

           参数:
               interface: 提供依赖项的 WBMpcInterface
               frequency: MPC 循环频率 (Hz)

           注意:
               此构造函数自动提取必要的组件
           )")

      // 线程管理
      .def("start_mpc_thread", &WBMpcMrtJointController::startMpcThread, py::arg("init_robot_state"), "启动后台 MPC 求解器线程")

      // 控制计算
      // 注意: compute_joint_control_action 的 actions 参数类型是 RobotJointAction& (基于 JointIdMap 的容器)
      // 而非 std::vector<JointAction>
      .def(
          "compute_joint_control_action",
          [](WBMpcMrtJointController& self, ocs2::scalar_t time, const RobotState& state, RobotJointAction& actions) {
            self.computeJointControlAction(time, state, actions);
            return actions;
          },
          py::arg("time"), py::arg("state"), py::arg("actions_out"),
          R"(
           计算关节控制动作。

           参数:
               time: 当前仿真时间
               state: 当前机器人状态
               actions_out: RobotJointAction 容器，用于输出动作

           返回:
               包含计算指令的修改后的动作容器
           )")

      // 状态
      .def("ready", &WBMpcMrtJointController::ready, "检查控制器是否就绪 (已收到策略)")

      .def("__repr__", [](const WBMpcMrtJointController& self) {
        return "<WBMpcMrtJointController ready=" + std::string(self.ready() ? "True" : "False") + ">";
      });

  // =============================================================================
  // 5. MpcWeightAdjustmentModule 绑定
  // =============================================================================

  py::class_<MpcWeightAdjustmentModule, std::shared_ptr<MpcWeightAdjustmentModule>>(m, "MpcWeightAdjustmentModule")
      .def(py::init<WBMpcInterface&>(), py::arg("interface"),
           R"(
           创建用于 RL 集成的权重调整模块。

           此模块与 MPC 求解器同步,
           根据 RL 策略输出调整代价函数权重。
           )")

      .def("set_residual_weights", &MpcWeightAdjustmentModule::setResidualWeights, py::arg("residual_actions"),
           R"(
           设置来自 RL 策略的残差权重。

           参数:
               residual_actions: 来自 RL 策略的 70 个权重值列表
           )")

      .def("__repr__", [](const MpcWeightAdjustmentModule& self) { return "<MpcWeightAdjustmentModule>"; });

  // =============================================================================
  // 6. MujocoSimInterface 绑定
  // =============================================================================

  py::class_<robot::mujoco_sim_interface::MujocoSimConfig>(m, "MujocoSimConfig")
      .def(py::init<>())
      .def_readwrite("scene_path", &robot::mujoco_sim_interface::MujocoSimConfig::scenePath)
      .def_readwrite("dt", &robot::mujoco_sim_interface::MujocoSimConfig::dt)
      .def_readwrite("render_frequency_hz", &robot::mujoco_sim_interface::MujocoSimConfig::renderFrequencyHz)
      .def_readwrite("headless", &robot::mujoco_sim_interface::MujocoSimConfig::headless)
      .def_readwrite("verbose", &robot::mujoco_sim_interface::MujocoSimConfig::verbose)
      .def(
          "set_init_state",
          [](robot::mujoco_sim_interface::MujocoSimConfig& self, const robot::model::RobotState& state) {
            self.initStatePtr_ = std::make_shared<robot::model::RobotState>(state);
          },
          py::arg("state"), "设置初始机器人状态");

  // 注意：不声明基类 RobotHWInterfaceBase，因为它在 Python 中未绑定
  // MujocoSimInterface 的方法已直接暴露，无需通过基类接口访问
  py::class_<robot::mujoco_sim_interface::MujocoSimInterface>(m, "MujocoSimInterface")
      .def(py::init<const robot::mujoco_sim_interface::MujocoSimConfig&, const std::string&>(), py::arg("config"), py::arg("urdf_path"),
           R"(
           创建 Mujoco 仿真接口。

           参数:
               config: 仿真配置
               urdf_path: 机器人 URDF 文件的路径
           )")

      // 仿真控制
      .def("init_sim", &robot::mujoco_sim_interface::MujocoSimInterface::initSim, "初始化仿真")
      .def("start_sim", &robot::mujoco_sim_interface::MujocoSimInterface::startSim, "启动仿真循环")
      .def("simulation_step", &robot::mujoco_sim_interface::MujocoSimInterface::simulationStep, "执行一个仿真步骤")
      .def("reset", &robot::mujoco_sim_interface::MujocoSimInterface::reset, "重置仿真")
      .def(
          "set_robot_state",
          [](robot::mujoco_sim_interface::MujocoSimInterface& self, const robot::model::RobotState& state) {
            // 通过 reset 然后设置状态来更新仿真器状态
            // 注意：这需要访问私有方法，可能需要添加公共接口
            self.reset();
            // 获取当前状态并更新（通过引用修改）
            robot::model::RobotState& current_state = const_cast<robot::model::RobotState&>(self.getRobotState());
            current_state = state;
            // 需要调用 setSimState，但这是私有方法
            // 暂时通过 reset 和手动设置来处理
          },
          py::arg("state"), "设置机器人状态（实验性）")

      // 状态访问
      .def("get_robot_state", &robot::mujoco_sim_interface::MujocoSimInterface::getRobotState, py::return_value_policy::reference,
           "获取当前机器人状态")
      .def("get_robot_joint_action", &robot::mujoco_sim_interface::MujocoSimInterface::getRobotJointAction,
           py::return_value_policy::reference, "获取关节动作缓冲区")
      .def("update_interface_state_from_robot", &robot::mujoco_sim_interface::MujocoSimInterface::updateInterfaceStateFromRobot,
           "从仿真器更新机器人状态到接口")
      .def("apply_joint_action", &robot::mujoco_sim_interface::MujocoSimInterface::applyJointAction, "将关节动作应用到仿真器")

      // 模型访问
      .def("get_model", &robot::mujoco_sim_interface::MujocoSimInterface::getModel, py::return_value_policy::reference,
           "获取 Mujoco 模型指针")

// 同步模式
#if SYNCHRONOUS_SIMULATION_MODE
      .def("step", &robot::mujoco_sim_interface::MujocoSimInterface::step, "执行一个仿真步骤 (同步模式)")
      .def("is_sim_initialized", &robot::mujoco_sim_interface::MujocoSimInterface::isSimInitialized, "检查仿真是否已初始化")
#endif

      .def("__repr__", [](const robot::mujoco_sim_interface::MujocoSimInterface& self) { return "<MujocoSimInterface>"; });

  // =============================================================================
  // 7. TargetTrajectories 绑定
  // =============================================================================

  py::class_<ocs2::TargetTrajectories>(m, "TargetTrajectories")
      .def(py::init<>())
      .def(py::init<ocs2::scalar_array_t, ocs2::vector_array_t, ocs2::vector_array_t>(), py::arg("time_trajectory"),
           py::arg("state_trajectory"), py::arg("input_trajectory") = ocs2::vector_array_t(),
           R"(
           创建目标轨迹对象。

           参数:
               time_trajectory: 时间轨迹 (标量数组)
               state_trajectory: 状态轨迹 (向量数组)
               input_trajectory: 输入轨迹 (向量数组，可选)
           )")
      .def_readwrite("time_trajectory", &ocs2::TargetTrajectories::timeTrajectory, "时间轨迹")
      .def_readwrite("state_trajectory", &ocs2::TargetTrajectories::stateTrajectory, "状态轨迹")
      .def_readwrite("input_trajectory", &ocs2::TargetTrajectories::inputTrajectory, "输入轨迹")
      .def("clear", &ocs2::TargetTrajectories::clear, "清空轨迹")
      .def("empty", &ocs2::TargetTrajectories::empty, "检查轨迹是否为空")
      .def("size", &ocs2::TargetTrajectories::size, "获取轨迹大小")
      .def("get_desired_state", &ocs2::TargetTrajectories::getDesiredState, py::arg("time"), "获取指定时间的期望状态")
      .def("get_desired_input", &ocs2::TargetTrajectories::getDesiredInput, py::arg("time"), "获取指定时间的期望输入")
      .def("__repr__",
           [](const ocs2::TargetTrajectories& self) { return "<TargetTrajectories size=" + std::to_string(self.size()) + ">"; });

  // =============================================================================
  // 8. WBMpcTargetTrajectoriesCalculator 绑定
  // =============================================================================

  py::class_<WBMpcTargetTrajectoriesCalculator>(m, "WBMpcTargetTrajectoriesCalculator")
      .def(py::init([](const std::string& reference_file, WBMpcInterface& interface) {
             // 从 WBMpcInterface 获取所需的依赖
             return new WBMpcTargetTrajectoriesCalculator(reference_file, interface.getMpcRobotModel(),
                                                          interface.mpcSettings().timeHorizon_);
           }),
           py::arg("reference_file"), py::arg("interface"),
           R"(
           创建目标轨迹计算器。

           参数:
               reference_file: 参考配置文件的路径
               interface: WBMpcInterface 实例，用于获取 MPC 机器人模型和时间范围

           使用示例:
               calculator = WBMpcTargetTrajectoriesCalculator(reference_file, mpc_interface)
               # 通过速度指令生成目标轨迹
               target_traj = calculator.commanded_velocity_to_target_trajectories(
                   commanded_velocities=[v_x, v_y, v_z, v_yaw],
                   init_time=0.0,
                   init_state=current_state
               )
           )")
      .def(
          "commanded_position_to_target_trajectories",
          [](WBMpcTargetTrajectoriesCalculator& self, const Eigen::Ref<const Eigen::Vector4d>& command_line_pose_target,
             ocs2::scalar_t init_time, const Eigen::Ref<const ocs2::vector_t>& init_state) {
            ocs2::vector4_t cmd;
            cmd << command_line_pose_target(0), command_line_pose_target(1), command_line_pose_target(2), command_line_pose_target(3);
            return self.commandedPositionToTargetTrajectories(cmd, init_time, init_state);
          },
          py::arg("command_line_pose_target"), py::arg("init_time"), py::arg("init_state"),
          R"(
           将位置指令转换为目标轨迹。

           参数:
               command_line_pose_target: 位置指令 [deltaX, deltaY, deltaZ, deltaYaw]，定义在骨盆坐标系中
               init_time: 初始时间
               init_state: 初始状态向量

           返回:
               TargetTrajectories 对象，包含生成的目标轨迹
           )")
      .def(
          "commanded_velocity_to_target_trajectories",
          [](WBMpcTargetTrajectoriesCalculator& self, const Eigen::Ref<const Eigen::Vector4d>& commanded_velocities,
             ocs2::scalar_t init_time, const Eigen::Ref<const ocs2::vector_t>& init_state) {
            ocs2::vector4_t cmd;
            cmd << commanded_velocities(0), commanded_velocities(1), commanded_velocities(2), commanded_velocities(3);
            return self.commandedVelocityToTargetTrajectories(cmd, init_time, init_state);
          },
          py::arg("commanded_velocities"), py::arg("init_time"), py::arg("init_state"),
          R"(
           将速度指令转换为目标轨迹。

           参数:
               commanded_velocities: 速度指令 [v_x, v_y, v_z, v_yaw]，定义在骨盆坐标系中
               init_time: 初始时间
               init_state: 初始状态向量

           返回:
               TargetTrajectories 对象，包含生成的目标轨迹

           注意:
               此方法会生成一个插值轨迹，在前半部分时间范围内从当前动量插值到期望动量，
               在后半部分时间范围内完全应用期望动量。所有位置目标都是通过对速度曲线积分得到的。
           )")
      .def("__repr__", [](const WBMpcTargetTrajectoriesCalculator& self) { return "<WBMpcTargetTrajectoriesCalculator>"; });

  // =============================================================================
  // 9. 工具函数
  // =============================================================================

  m.def("get_mpc_joint_dim_from_model", &WBMpcInterface::getInputDim, "获取 MPC 输入维度 (辅助函数)");

  // =============================================================================
  // 10. 枚举和常量
  // =============================================================================

  // 如果代码库中有枚举,在这里绑定它们
  // 示例:
  // py::enum_<ContactState>(m, "ContactState")
  //     .value("NO_CONTACT", ContactState::NO_CONTACT)
  //     .value("CONTACT", ContactState::CONTACT)
  //     .export_values();
}
