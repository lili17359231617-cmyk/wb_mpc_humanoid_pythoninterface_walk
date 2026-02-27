/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************************************************************/

/**
 * @file WBMpcMrtJointController.cpp
 * @brief MPC控制输出到电机输出的转换实现
 *
 * @section MPC控制流程概述
 *
 * 本控制器实现了MPC（模型预测控制）输出到机器人电机控制的转换。MPC优化的输出包括：
 *   1. 期望轨迹状态 (mpcPolicyState): 包含期望的关节角度、速度、基座位置/姿态等
 *   2. 控制输入 (mpcPolicyInput): 包含足端接触力/力矩、关节加速度
 *   3. 接触模式 (mpcPolicyMode): 指示哪些脚处于支撑相
 *
 * @section 控制量转换流程
 *
 * MPC输出 -> 电机输出的转换遵循以下步骤：
 *
 *   Step 1: 获取MPC策略输出
 *           - 从MPC获取期望状态和控制输入
 *
 *   Step 2: 逆动力学计算关节力矩
 *           - 输入: 足端力/力矩 (W_left, W_right) + 关节加速度 (qdd_joints)
 *           - 处理: 使用Pinocchio计算质量矩阵M、非线性项nle、足端雅可比矩阵J
 *           - 公式: τ_joints = M * qdd + nle - J^T * F_ext
 *           - 输出: 各关节需要输出的力矩 τ_joints
 *
 *   Step 3: 提取期望轨迹
 *           - 从MPC状态中提取期望关节角度 q_des
 *           - 从MPC状态中提取期望关节速度 qd_des
 *
 *   Step 4: 生成电机控制指令
 *           - 使用PD控制律 + 前馈力矩
 *           - 最终输出: motor_torque = kp * (q_des - q_current) + kd * (qd_des - qd_current) + feed_forward_effort
 *
 * @section 输入输出向量定义
 *
 * 状态向量 x = [p_base, euler, q_joints, v_base, euler_d, qd_joints]^T
 *   - p_base (3): 基座在世界坐标系下的位置
 *   - euler (3): 基座姿态的ZYX欧拉角
 *   - q_joints (n): 关节角度
 *   - v_base (3): 基座在世界坐标系下的线速度
 *   - euler_d (3): 欧拉角导数
 *   - qd_joints (n): 关节速度
 *
 * 输入向量 u = [W_l, W_r, qdd_joints]^T
 *   - W_l (6): 左脚末端接触力/力矩 [f_x, f_y, f_z, M_x, M_y, M_z]
 *   - W_r (6): 右脚末端接触力/力矩 [f_x, f_y, f_z, M_x, M_y, M_z]
 *   - qdd_joints (n): 关节加速度
 *
 * 电机控制指令 = [q_des, qd_des, kp, kd, feed_forward_effort]
 *   - q_des: 期望关节角度
 *   - qd_des: 期望关节速度
 *   - kp: 位置环比例增益
 *   - kd: 速度环微分增益
 *   - feed_forward_effort: 前馈力矩（来自逆动力学计算）
 */

#include "humanoid_wb_mpc/mrt/WBMpcMrtJointController.h"

#include <ocs2_robotic_tools/common/RotationDerivativesTransforms.h>
#include <ocs2_robotic_tools/common/RotationTransforms.h>

#include <humanoid_common_mpc/gait/MotionPhaseDefinition.h>
#include <humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h>
#include <humanoid_common_mpc/reference_manager/ProceduralMpcMotionManager.h>
#include "humanoid_wb_mpc/dynamics/DynamicsHelperFunctions.h"

namespace ocs2::humanoid {

WBMpcMrtJointController::WBMpcMrtJointController(const ::robot::model::RobotDescription& robotDescription,
                                                 const ModelSettings& modelSettings,
                                                 MPC_BASE& mpc,
                                                 PinocchioInterface pinocchioInterface,
                                                 scalar_t mpcDesiredFrequency,
                                                 std::shared_ptr<DummyObserver> rVizVisualizerPtr)
    : mcpMrtInterface_(mpc),
      pinocchioInterface_(pinocchioInterface),
      mpcRobotModel_(modelSettings),
      mpcDeltaTMicroSeconds_(1000000 / mpcDesiredFrequency),
      realtime_(mpcDesiredFrequency <= 0),
      visualizerPtr_(rVizVisualizerPtr) {
  mpcJointIndices_ = robotDescription.getJointIndices(modelSettings.mpcModelJointNames);
  otherJointIndices_ = robotDescription.getJointIndices(modelSettings.fixedJointNames);
  currentMpcObservation_.state = vector_t::Zero(mpcRobotModel_.getStateDim());
  currentMpcObservation_.input = vector_t::Zero(mpcRobotModel_.getInputDim());
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

WBMpcMrtJointController::~WBMpcMrtJointController() {
  // Signal the solver thread to terminate
  terminateThread_.store(true);

  // Wait for the solver thread to finish if it's joinable
  if (solver_worker_.joinable()) {
    solver_worker_.join();
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void WBMpcMrtJointController::startMpcThread(const ::robot::model::RobotState& initRobotState) {
  updateMpcObservation(currentMpcObservation_, initRobotState);
  // Set observation to MPC
  mcpMrtInterface_.setCurrentObservation(currentMpcObservation_);
  solver_worker_ = std::jthread(&WBMpcMrtJointController::solverWorker, this);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void WBMpcMrtJointController::updateMpcState(vector_t& mpcState, const ::robot::model::RobotState& robotState) {
  mpcRobotModel_.setBasePosition(mpcState, robotState.getRootPositionInWorldFrame());
  mpcRobotModel_.setBaseOrientationEulerZYX(mpcState, quaternionToEulerZYX(robotState.getRootRotationLocalToWorldFrame()));

  mpcRobotModel_.setJointAngles(mpcState, robotState.getJointPositions(mpcJointIndices_));

  // currently we send local angular and linear velocity
  mpcRobotModel_.setBaseLinearVelocity(mpcState,
                                       robotState.getRootRotationLocalToWorldFrame() * robotState.getRootLinearVelocityInLocalFrame());
  mpcRobotModel_.setBaseOrientationEulerZYXDerivatives(
      mpcState, getEulerAnglesZyxDerivativesFromLocalAngularVelocity<scalar_t>(mpcRobotModel_.getBaseOrientationEulerZYX(mpcState),
                                                                               robotState.getRootAngularVelocityInLocalFrame()));

  vector_t dummyInput = vector_t::Zero(mpcRobotModel_.getInputDim());
  mpcRobotModel_.setJointVelocities(mpcState, dummyInput, robotState.getJointVelocities(mpcJointIndices_));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void WBMpcMrtJointController::updateMpcObservation(ocs2::SystemObservation& mpcObservation, const ::robot::model::RobotState& robotState) {
  updateMpcState(mpcObservation.state, robotState);
  mpcObservation.time = robotState.getTime();
  mpcObservation.input = vector_t::Zero(mpcRobotModel_.getInputDim());  // Add contact forces later.
  std::vector<bool> configContacts = robotState.getContactFlags();
  assert(configContacts.size() == 2);
  contact_flag_t contactFlags;
  std::copy(configContacts.begin(), configContacts.end(), contactFlags.begin());
  mpcObservation.mode = stanceLeg2ModeNumber(contactFlags);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

/**
 * @brief 计算关节控制动作 - MPC输出到电机输出的核心转换函数
 *
 * @details
 * 此函数是MPC控制器与电机执行器之间的桥梁，负责将MPC优化结果转换为电机控制指令。
 *
 * 转换流程详解：
 * ┌─────────────────────────────────────────────────────────────────────────────┐
 * │                        MPC输出到电机输出转换流程                              │
 * ├─────────────────────────────────────────────────────────────────────────────┤
 * │                                                                              │
 * │   ┌──────────────────┐     ┌──────────────────────────────────────────┐     │
 * │   │   MPC 优化结果    │     │                                          │     │
 * │   │                  │     │  Step 1: 获取MPC策略输出                    │     │
 * │   │ • mpcPolicyState │────▶│    - 状态: 位置/速度/姿态                   │     │
 * │   │ • mpcPolicyInput │     │    - 输入: 足端力/力矩 + 关节加速度          │     │
 * │   │ • mpcPolicyMode  │     │    - 模式: 支撑相信息                       │     │
 * │   └──────────────────┘     └──────────────────────────────────────────┘     │
 * │                                  │                                          │
 * │                                  ▼                                          │
 * │   ┌──────────────────────────────────────────────────────────────────────┐  │
 * │   │              Step 2: 逆动力学计算关节力矩                              │  │
 * │   │                                                                       │  │
 * │   │   输入:                                                                │  │
 * │   │     • 足端接触力/力矩 W_left, W_right (6维)                            │  │
 * │   │     • 关节加速度 qdd_joints                                           │  │
 * │   │     • 关节位置 q、速度 qd                                             │  │
 * │   │                                                                       │  │
 * │   │   计算过程:                                                            │  │
 * │   │     1. 计算质量矩阵 M = CRBA(q)                                       │  │
 * │   │     2. 计算非线性项 nle = NonLinearEffects(q, qd)                     │  │
 * │   │     3. 计算足端雅可比矩阵 J_left, J_right                             │  │
 * │   │     4. 投影足端力到关节空间: F_joint = J^T * W                         │  │
 * │   │     5. 逆动力学: τ = M * qdd + nle - F_joint                          │  │
 * │   │                                                                       │  │
 * │   │   物理意义:                                                            │  │
 * │   │     τ = 惯性力 + 重力/科氏力/离心力 - 外力引起的力                      │  │
 * │   └──────────────────────────────────────────────────────────────────────┘  │
 * │                                  │                                          │
 * │                                  ▼                                          │
 * │   ┌──────────────────────────────────────────────────────────────────────┐  │
 * │   │              Step 3: 提取期望轨迹                                      │  │
 * │   │                                                                       │  │
 * │   │   • mpc_q_desired  = 从MPC状态中提取关节角度                           │  │
 * │   │   • mpc_qd_desired = 从MPC状态中提取关节速度                           │  │
 * │   └──────────────────────────────────────────────────────────────────────┘  │
 * │                                  │                                          │
 * │                                  ▼                                          │
 * │   ┌──────────────────────────────────────────────────────────────────────┐  │
 * │   │              Step 4: 生成电机控制指令 (PD + 前馈)                      │  │
 * │   │                                                                       │  │
 * │   │   最终电机力矩 = FeedForward + Kp * (q_des - q) + Kd * (qd_des - qd)   │  │
 * │   │                                                                       │  │
 * │   │   控制器参数:                                                          │  │
 * │   │     • kp = 1200.0  (位置环比例增益)                                   │  │
 * │   │     • kd = 10.0    (速度环微分增益)                                   │  │
 * │   │     • feed_forward_effort = 逆动力学计算的关节力矩                      │  │
 * │   └──────────────────────────────────────────────────────────────────────┘  │
 * │                                  │                                          │
 * │                                  ▼                                          │
 * │   ┌──────────────────┐                                                         │
 * │   │   电机控制指令     │                                                         │
 * │   │   RobotJointAction │◀──── 写入到各关节控制结构                           │
 * │   │                  │     - q_des, qd_des, kp, kd, feed_forward_effort    │
 * │   └──────────────────┘                                                         │
 * │                                                                              │
 * └─────────────────────────────────────────────────────────────────────────────┘
 *
 * @param [in] time 当前仿真时间
 * @param [in] robotState 机器人当前状态（传感器反馈）
 * @param [out] robotJointAction 输出到执行器的电机控制指令
 */
void WBMpcMrtJointController::computeJointControlAction(scalar_t time,
                                                        const ::robot::model::RobotState& robotState,
                                                        ::robot::model::RobotJointAction& robotJointAction) {
  updateMpcObservation(currentMpcObservation_, robotState);
  // Set observation to MPC
  mcpMrtInterface_.setCurrentObservation(currentMpcObservation_);

  vector_t mpcPolicyState;
  vector_t mpcPolicyInput;
  size_t mpcPolicyMode;

  if (mcpMrtInterface_.initialPolicyReceived()) {
    // Evaluate policy with feedback if activated in config
    mcpMrtInterface_.evaluatePolicy(currentMpcObservation_.time + 0.005, currentMpcObservation_.state, mpcPolicyState, mpcPolicyInput,
                                    mpcPolicyMode);

    vector_t mpcJointTorques = computeJointTorques<scalar_t>(mpcPolicyState, mpcPolicyInput, pinocchioInterface_, mpcRobotModel_);
    vector_t mpc_q_desired = mpcRobotModel_.getJointAngles(mpcPolicyState);
    vector_t mpc_qd_desired = mpcRobotModel_.getJointVelocities(mpcPolicyState, mpcPolicyInput);

    // std::cout << "mpcJointTorques: " << mpcJointTorques.transpose() << std::endl;

    for (size_t i = 0; i < mpcJointIndices_.size(); i++) {
      size_t index = mpcJointIndices_[i];
      robot::model::JointAction& action = robotJointAction.at(index).value();

      action.q_des = mpc_q_desired[i];
      action.qd_des = mpc_qd_desired[i];
      action.kp = 1200.0;
      action.kd = 10.0;
      action.feed_forward_effort = mpcJointTorques[i];

      // std::cerr << "MPCtorque!: " << mpcJointTorques[i] << std::endl;
    };

    if (visualizerPtr_ != nullptr) {
      visualizerPtr_->update(currentMpcObservation_, mcpMrtInterface_.getPolicy(), mcpMrtInterface_.getCommand());
    }
  }

  else {
    std::cerr << "Apply weight compensating torque..." << std::endl;
    //   Apply weight compensated input around current state
    mpcPolicyState = currentMpcObservation_.state;
    mpcPolicyInput = weightCompensatingInput(pinocchioInterface_, {true, true}, mpcRobotModel_);
    vector_t weightCompensatingTorques = computeJointTorques<scalar_t>(mpcPolicyState, mpcPolicyInput, pinocchioInterface_, mpcRobotModel_);

    for (size_t i = 0; i < mpcJointIndices_.size(); i++) {
      size_t index = mpcJointIndices_[i];
      robot::model::JointAction& action = robotJointAction.at(index).value();

      action.q_des = 0;
      action.qd_des = 0;
      action.kp = 0;
      action.kd = 0;
      action.feed_forward_effort = weightCompensatingTorques[i];
    };
  }

  for (size_t i = 0; i < otherJointIndices_.size(); i++) {
    size_t index = otherJointIndices_[i];
    robot::model::JointAction& action = robotJointAction.at(index).value();

    action.q_des = 0;
    action.qd_des = 0;
    action.kp = 100;
    action.kd = 1.0;
    action.feed_forward_effort = 0.0;
  };
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void WBMpcMrtJointController::solverWorker() {
  mcpMrtInterface_.resetMpcNode(currentObservationToResetTrajectory(mcpMrtInterface_.getCurrentObservation()));
  std::cerr << "MPC is reset. NMPC solver started!" << std::endl;

  while (true) {
    auto targetTimeForNextIteration = std::chrono::steady_clock::now() + std::chrono::microseconds(mpcDeltaTMicroSeconds_);

    mcpMrtInterface_.advanceMpc();

    // Publish if Policy has been updated
    if (!mcpMrtInterface_.updatePolicy()) {
      std::cerr << "The solver has failed to update!!" << std::endl;
      return;
    }

    // std::cerr << "MPC policy computed!" << std::endl;

    if (!realtime_) {
      auto currentTime = std::chrono::steady_clock::now();
      if (currentTime > targetTimeForNextIteration) {
        auto delay = std::chrono::duration_cast<std::chrono::microseconds>(currentTime - targetTimeForNextIteration).count();

        std::cerr << "Warning: MPC loop running slow by " << delay << " microseconds." << std::endl;
      } else {
        // Sleep in case sim loop is faster than specified
        std::this_thread::sleep_until(targetTimeForNextIteration);
      }
    }
  }
  std::cerr << "Shutting down NMPC" << std::endl;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
TargetTrajectories WBMpcMrtJointController::currentObservationToResetTrajectory(const SystemObservation& currentObservation) {
  vector_t targetState = currentObservation.state;

  // zero out velocities
  targetState.tail(mpcRobotModel_.getGenCoordinatesDim()) = vector_t::Zero(mpcRobotModel_.getGenCoordinatesDim());

  // zero out pitch + roll angles
  targetState.segment<2>(4) = vector_t::Zero(2);

  const TargetTrajectories resetTargetTrajectories({currentObservation.time}, {targetState},
                                                   {vector_t::Zero(currentObservation.input.size())});

  std::cerr << "Resetting MPC to current state: \n" << targetState << std::endl;
  return resetTargetTrajectories;
}

}  // namespace ocs2::humanoid
