/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.
Copyright (c) 2024, 1X Technologies. All rights reserved.

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

#pragma once

#include <ocs2_core/Types.h>
#include <ocs2_core/penalties/Penalties.h>
#include <ocs2_ddp/DDP_Settings.h>
#include <ocs2_mpc/MPC_BASE.h>           //引入 MPC 基础类头文件
#include <ocs2_mpc/MPC_MRT_Interface.h>  //引入 MPC MRT 接口头文件,用于获取控制结果
#include <ocs2_mpc/MPC_Settings.h>
#include <ocs2_oc/rollout/TimeTriggeredRollout.h>
#include <ocs2_oc/synchronized_module/SolverSynchronizedModule.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include <ocs2_robotic_tools/common/RobotInterface.h>
#include <ocs2_sqp/SqpSettings.h>

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/initialization/WeightCompInitializer.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_wb_mpc/common/WBAccelMpcRobotModel.h"
#include "humanoid_wb_mpc/end_effector/EndEffectorDynamics.h"

namespace ocs2::humanoid {
// 前向声明权重同步模块类
class MpcWeightAdjustmentModule;

class WBMpcInterface final : public RobotInterface {
 public:
  /**
   * Constructor
   *
   * @throw Invalid argument error if input task file or urdf file does not exist.
   *
   * @param [in] taskFile: The absolute path to the configuration file for the MPC.
   * @param [in] urdfFile: The absolute path to the URDF file for the robot.
   * @param [in] referenceFile: The absolute path to the reference configuration file.
   */
  WBMpcInterface(const std::string& taskFile, const std::string& urdfFile, const std::string& referenceFile, bool setupOCP = true);

  ~WBMpcInterface() override = default;

  /**
   * 核心封装：初始化 MPC 求解器
   * 建议在构造函数 setupOCP = true 的逻辑后调用
   */
  void setupMpc();
  /**
   * 核心封装：执行一步 MPC 并返回当前最优控制指令
   * @param observation 当前机器人状态 (q, v)
   * @param time 当前仿真时间
   * @return 最优控制输入 (对于该项目通常是关节加速度或力矩)
   */
  vector_t runMpc(const vector_t& observation, scalar_t time);
  /**
   * Reset MPC and MRT internal state (exposed to Python).
   */
  void reset();

  // 供 Python 获取底层求解器指针（用于挂载权重同步模块）
  MPC_BASE* getMpcPtr() { return mpcPtr_.get(); }

  /**
   * 将 SolverSynchronizedModule 挂到 MPC 求解器（与 C++ WBMpcRobotSim 一致）。
   * 须在 setupMpc() 之后调用；求解器每次求解前会调用该模块的 preSolverRun(initTime, finalTime, initState)。
   */
  void addSynchronizedModule(std::shared_ptr<ocs2::SolverSynchronizedModule> module);

  const OptimalControlProblem& getOptimalControlProblem() const override { return *problemPtr_; }

  const ModelSettings& modelSettings() const { return modelSettings_; }
  const ddp::Settings& ddpSettings() const { return ddpSettings_; }
  const mpc::Settings& mpcSettings() const { return mpcSettings_; }
  const rollout::Settings& rolloutSettings() const { return rolloutSettings_; }
  const sqp::Settings& sqpSettings() { return sqpSettings_; }

  const std::string& getTaskFile() const { return taskFile_; }
  const std::string& getURDFFile() const { return urdfFile_; }
  const std::string& getReferenceFile() const { return referenceFile_; }

  const vector_t& getInitialState() const { return initialState_; }
  const RolloutBase& getRollout() const { return *rolloutPtr_; }
  PinocchioInterface& getPinocchioInterface() { return *pinocchioInterfacePtr_; }
  std::shared_ptr<SwitchedModelReferenceManager> getSwitchedModelReferenceManagerPtr() const { return referenceManagerPtr_; }

  const WeightCompInitializer& getInitializer() const override { return *initializerPtr_; }
  std::shared_ptr<ReferenceManagerInterface> getReferenceManagerPtr() const override { return referenceManagerPtr_; }

  const WBAccelMpcRobotModel<scalar_t>& getMpcRobotModel() const { return *mpcRobotModelPtr_; }
  const WBAccelMpcRobotModel<ad_scalar_t>& getMpcRobotModelAD() const { return *mpcRobotModelADPtr_; }

  /** OCS2 状态维度，供 Python 校验观测向量长度。格式: [base_pos(3), base_euler(3), joint_pos(23), base_vel(3), euler_dot(3), joint_vel(23)]
   */
  size_t getStateDim() const { return mpcRobotModelPtr_->getStateDim(); }
  /** MPC 输入维度（接触力+关节加速度），供 Python 校验控制输出长度。 */
  size_t getInputDim() const { return mpcRobotModelPtr_->getInputDim(); }

  /// 增加获取该模块的接口（如果 Python 绑定需要）
  std::shared_ptr<MpcWeightAdjustmentModule> getWeightAdjustmentModule();
  // 使用专用的同步模块来管理权重
  /**
   * 供 Python 或其他模块调用，将 RL 输出的权重更新到 MPC 内部
   * @param newWeights Eigen 格式的残差向量
   */
  void updateMpcWeights(const ocs2::vector_t& newWeights);
  void setTargetState(const vector_t& targetState);
  void setTargetTrajectories(const TargetTrajectories& targetTrajectories);

 private:
  void setupOptimalControlProblem();

  std::unique_ptr<StateInputConstraint> getStanceFootConstraint(const EndEffectorDynamics<scalar_t>& eeDynamics, size_t contactPointIndex);
  std::unique_ptr<StateInputConstraint> getNormalVelocityConstraint(const EndEffectorDynamics<scalar_t>& eeDynamics,
                                                                    size_t contactPointIndex);
  std::unique_ptr<StateInputCost> getJointTorqueCost(const std::string& taskFile);
  std::unique_ptr<StateInputConstraint> getJointMimicConstraint(size_t mimicIndex);

  ModelSettings modelSettings_;
  ddp::Settings ddpSettings_;
  mpc::Settings mpcSettings_;
  sqp::Settings sqpSettings_;

  std::unique_ptr<PinocchioInterface> pinocchioInterfacePtr_;

  std::unique_ptr<OptimalControlProblem> problemPtr_;
  std::shared_ptr<SwitchedModelReferenceManager> referenceManagerPtr_;

  std::unique_ptr<WBAccelMpcRobotModel<scalar_t>> mpcRobotModelPtr_;
  std::unique_ptr<WBAccelMpcRobotModel<ad_scalar_t>> mpcRobotModelADPtr_;

  rollout::Settings rolloutSettings_;
  std::unique_ptr<RolloutBase> rolloutPtr_;
  std::unique_ptr<WeightCompInitializer> initializerPtr_;

  // 新增：求解器和 MRT 接口
  std::unique_ptr<MPC_BASE> mpcPtr_;                                      // MPC 求解器指针
  std::unique_ptr<MPC_MRT_Interface> mpcMrtPtr_;                          // MPC MRT 接口指针
  std::shared_ptr<MpcWeightAdjustmentModule> weightAdjustmentModulePtr_;  // 权重同步模块指针  // 权重同步模块指针2

  vector_t initialState_;

  const std::string taskFile_;
  const std::string urdfFile_;
  const std::string referenceFile_;
  bool verbose_;
};

}  // namespace ocs2::humanoid
