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

#include <iostream>
#include <stdexcept>
#include <string>

// Pinocchio forward declarations must be included first
#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include "humanoid_wb_mpc/WBMpcInterface.h"

#include <ocs2_core/misc/Display.h>
#include <ocs2_core/misc/LoadData.h>
#include <ocs2_core/misc/Numerics.h>
#include <ocs2_core/penalties/Penalties.h>
#include <ocs2_core/soft_constraint/StateInputSoftConstraint.h>
#include <ocs2_oc/synchronized_module/SolverSynchronizedModule.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>
#include <ocs2_sqp/SqpMpc.h>

#include <humanoid_common_mpc/pinocchio_model/createPinocchioModel.h>
#include "humanoid_common_mpc/HumanoidCostConstraintFactory.h"
#include "humanoid_common_mpc/initialization/WeightCompInitializer.h"

#include "humanoid_wb_mpc/WBMpcPreComputation.h"
#include "humanoid_wb_mpc/constraint/JointMimicDynamicsConstraint.h"
#include "humanoid_wb_mpc/constraint/SwingLegVerticalConstraintCppAd.h"
#include "humanoid_wb_mpc/constraint/ZeroAccelerationConstraintCppAd.h"
#include "humanoid_wb_mpc/cost/EndEffectorDynamicsFootCost.h"
#include "humanoid_wb_mpc/cost/JointTorqueCostCppAd.h"
#include "humanoid_wb_mpc/dynamics/WBAccelDynamicsAD.h"
#include "humanoid_wb_mpc/end_effector/PinocchioEndEffectorDynamicsCppAd.h"
#include "humanoid_wb_mpc/synchronized_module/MpcWeightAdjustmentModule.h"

// Boost
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

namespace ocs2::humanoid {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
// 资源初始化
WBMpcInterface::WBMpcInterface(const std::string& taskFile, const std::string& urdfFile, const std::string& referenceFile, bool setupOCP)
    : modelSettings_(taskFile, urdfFile, "wb_mpc_", "true"), taskFile_(taskFile), urdfFile_(urdfFile), referenceFile_(referenceFile) {
  // check that task file exists 确保 taskFile（任务参数）、urdfFile（机器人模型）和 referenceFile（步态参考）都存在
  boost::filesystem::path taskFilePath(taskFile);
  if (boost::filesystem::exists(taskFilePath)) {
    std::cerr << "[WBMpcInterface] Loading task file: " << taskFilePath << std::endl;
  } else {
    throw std::invalid_argument("[WBMpcInterface] Task file not found: " + taskFilePath.string());
  }
  // check that urdf file exists
  boost::filesystem::path urdfFilePath(urdfFile);
  if (boost::filesystem::exists(urdfFilePath)) {
    std::cerr << "[WBMpcInterface] Loading Pinocchio model from: " << urdfFilePath << std::endl;
  } else {
    throw std::invalid_argument("[WBMpcInterface] URDF file not found: " + urdfFilePath.string());
  }
  // check that targetCommand file exists
  boost::filesystem::path referenceFilePath(referenceFile);
  if (boost::filesystem::exists(referenceFilePath)) {
    std::cerr << "[WBMpcInterface] Loading target command settings from: " << referenceFilePath << std::endl;
  } else {
    throw std::invalid_argument("[WBMpcInterface] targetCommand file not found: " + referenceFilePath.string());
  }

  loadData::loadCppDataType(taskFile, "interface.verbose", verbose_);

  // load setting from loading file 通过 loadSettings 函数族载入 DDP、MPC、SQP 等算法的具体求解参数（例如预测步长、迭代次数）
  ddpSettings_ = ddp::loadSettings(taskFile, "ddp", verbose_);
  mpcSettings_ = mpc::loadSettings(taskFile, "mpc", verbose_);
  rolloutSettings_ = rollout::loadSettings(taskFile, "rollout", verbose_);
  sqpSettings_ = sqp::loadSettings(taskFile, "multiple_shooting", verbose_);

  // PinocchioInterface 创建机器人的刚体动力学内核
  pinocchioInterfacePtr_.reset(new PinocchioInterface(createCustomPinocchioInterface(taskFile, urdfFile, modelSettings_)));

  // Setup WB State Input Mapping 创建 x 和 u 的映射模型（普通版用于计算，AD 版用于自动微分）
  mpcRobotModelPtr_.reset(new WBAccelMpcRobotModel<scalar_t>(modelSettings_));
  mpcRobotModelADPtr_.reset(new WBAccelMpcRobotModel<ad_scalar_t>(modelSettings_));

  // Swing trajectory planner 管理摆动腿轨迹
  std::unique_ptr<SwingTrajectoryPlanner> swingTrajectoryPlanner(
      new SwingTrajectoryPlanner(loadSwingTrajectorySettings(taskFile, "swing_trajectory_config", verbose_), N_CONTACTS));

  // Mode schedule manager 管理步态
  referenceManagerPtr_ =
      std::make_shared<SwitchedModelReferenceManager>(GaitSchedule::loadGaitSchedule(referenceFile, modelSettings_, verbose_),
                                                      std::move(swingTrajectoryPlanner), *pinocchioInterfacePtr_, *mpcRobotModelPtr_);
  referenceManagerPtr_->setArmSwingReferenceActive(true);

  // initial state
  initialState_.setZero(mpcRobotModelPtr_->getStateDim());
  loadData::loadEigenMatrix(taskFile, "initialState", initialState_);

  if (setupOCP) {
    setupOptimalControlProblem();
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
// Reset MPC/MRT internal state（供 python 调用）
void WBMpcInterface::reset() {
  if (mpcPtr_) {
    mpcPtr_->reset();
  }
  if (mpcMrtPtr_) {
    mpcMrtPtr_->reset();
  }
}

// 建立最优控制问题
void WBMpcInterface::setupOptimalControlProblem() {
  HumanoidCostConstraintFactory factory =
      HumanoidCostConstraintFactory(taskFile_, referenceFile_, *referenceManagerPtr_, *pinocchioInterfacePtr_, *mpcRobotModelPtr_,
                                    *mpcRobotModelADPtr_, modelSettings_, verbose_);

  // Optimal control problem
  problemPtr_.reset(new OptimalControlProblem);

  // Dynamics 系统动力学模型，MPC 预测未来状态的依据
  std::unique_ptr<SystemDynamicsBase> dynamicsPtr;
  const std::string modelName = "dynamics";
  dynamicsPtr.reset(new WBAccelDynamicsAD(*pinocchioInterfacePtr_, *mpcRobotModelADPtr_, modelName, modelSettings_));

  problemPtr_->dynamicsPtr = std::move(dynamicsPtr);

  // Cost terms ， 对应 Q 和 R 矩阵
  problemPtr_->costPtr->add("stateInputQuadraticCost", factory.getStateInputQuadraticCost());
  // problemPtr_->costPtr->add("jointTorqueCost", getJointTorqueCost(taskFile_));
  problemPtr_->finalCostPtr->add("terminalCost", factory.getTerminalCost());

  // Constraints 软约束：关节限位、足端碰撞
  problemPtr_->stateSoftConstraintPtr->add("jointLimits", factory.getJointLimitsConstraint());
  problemPtr_->stateSoftConstraintPtr->add("FootCollisionSoftConstraint", factory.getFootCollisionConstraint());
  // Constraint terms

  // 足端追踪，让机器人能按照预定的轨迹抬腿
  EndEffectorDynamicsWeights footTrackingCostWeights =
      EndEffectorDynamicsWeights::getWeights(taskFile_, "task_space_foot_cost_weights.", verbose_);

  // check for mimic joints
  boost::property_tree::ptree pt;
  boost::property_tree::read_info(taskFile_, pt);
  bool hasMimicJoints = loadData::containsPtreeValueFind(pt, "mimicJoints");

  // 软约束：摩擦锥（frictionForceCone）防止打滑，接触力矩（contactMomentXY）防止脚底翻转
  // 硬约束：足端零加速度、零速度等
  for (size_t i = 0; i < N_CONTACTS; i++) {
    const std::string& footName = modelSettings_.contactNames[i];

    std::unique_ptr<EndEffectorDynamics<scalar_t>> eeDynamicsPtr;
    eeDynamicsPtr.reset(new PinocchioEndEffectorDynamicsCppAd(*pinocchioInterfacePtr_, *mpcRobotModelADPtr_, {footName}, footName,
                                                              modelSettings_.modelFolderCppAd, modelSettings_.recompileLibrariesCppAd,
                                                              modelSettings_.verboseCppAd));

    problemPtr_->softConstraintPtr->add(footName + "_frictionForceCone", factory.getFrictionForceConeConstraint(i));
    problemPtr_->softConstraintPtr->add(footName + "_contactMomentXY",
                                        factory.getContactMomentXYConstraint(i, footName + "_contact_moment_XY_constraint"));
    problemPtr_->equalityConstraintPtr->add(footName + "_zeroWrench", factory.getZeroWrenchConstraint(i));
    problemPtr_->equalityConstraintPtr->add(footName + "_zeroVelocity", getStanceFootConstraint(*eeDynamicsPtr, i));
    problemPtr_->equalityConstraintPtr->add(footName + "_normalVelocity", getNormalVelocityConstraint(*eeDynamicsPtr, i));

    if (hasMimicJoints) {
      problemPtr_->equalityConstraintPtr->add(footName + "_kneeJointMimic", getJointMimicConstraint(i));
    }

    std::string footTrackingCostName = footName + "_TaskSpaceTrackingCost";

    problemPtr_->costPtr->add(footTrackingCostName, std::unique_ptr<StateInputCost>(new EndEffectorDynamicsFootCost(
                                                        *referenceManagerPtr_, footTrackingCostWeights, *pinocchioInterfacePtr_,
                                                        *eeDynamicsPtr, *mpcRobotModelADPtr_, i, footTrackingCostName, modelSettings_)));
  }

  // Pre-computation 在求解开始前先算好所有脚的雅可比矩阵，提高频率
  problemPtr_->preComputationPtr.reset(
      new WBMpcPreComputation(*pinocchioInterfacePtr_, *referenceManagerPtr_->getSwingTrajectoryPlanner(), *mpcRobotModelPtr_));

  // Rollout 用于生成预测轨迹
  rolloutPtr_.reset(new TimeTriggeredRollout(*problemPtr_->dynamicsPtr, rolloutSettings_));

  // Initialization 利用重力补偿给出一个初始猜测，让 MPC 第一步就能站稳
  initializerPtr_.reset(new WeightCompInitializer(*pinocchioInterfacePtr_, *referenceManagerPtr_, *mpcRobotModelPtr_));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

// 支撑腿约束：通过 PD 控制（位置误差 + 速度误差）反馈到加速度级，保证了机器人站立时的稳固性
std::unique_ptr<StateInputConstraint> WBMpcInterface::getStanceFootConstraint(const EndEffectorDynamics<scalar_t>& eeDynamics,
                                                                              size_t contactPointIndex) {
  const ModelSettings::FootConstraintConfig& footCfg = modelSettings_.footConstraintConfig;

  EndEffectorDynamicsAccelerationsConstraint::Config config;
  config.b.setZero(6);
  config.Ax.setZero(6, 6);
  config.Av.setIdentity(6, 6);
  config.Aa.setIdentity(6, 6);
  if (!numerics::almost_eq(footCfg.positionErrorGain_z, 0.0)) {
    config.Ax(2, 2) = footCfg.positionErrorGain_z;
  }
  if (!numerics::almost_eq(footCfg.orientationErrorGain, 0.0)) {
    config.Ax.block(3, 3, 3, 3) = Eigen::MatrixXd::Identity(3, 3) * footCfg.orientationErrorGain;
  }
  config.Av.block(0, 0, 2, 2) = Eigen::MatrixXd::Identity(2, 2) * footCfg.linearVelocityErrorGain_xy;
  config.Av(2, 2) = footCfg.linearVelocityErrorGain_z;
  config.Av.block(3, 3, 3, 3) = Eigen::MatrixXd::Identity(3, 3) * footCfg.angularVelocityErrorGain;
  config.Aa.block(0, 0, 2, 2) = Eigen::MatrixXd::Identity(2, 2) * footCfg.linearAccelerationErrorGain_xy;
  config.Aa(2, 2) = footCfg.linearAccelerationErrorGain_z;
  config.Aa.block(3, 3, 3, 3) = Eigen::MatrixXd::Identity(3, 3) * footCfg.angularAccelerationErrorGain;

  return std::unique_ptr<StateInputConstraint>(
      new ZeroAccelerationConstraintCppAd(*referenceManagerPtr_, eeDynamics, contactPointIndex, config));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

// 关节联动约束：保证了 MPC 规划出的全身动作符合机器人的机械结构限制
std::unique_ptr<StateInputConstraint> WBMpcInterface::getJointMimicConstraint(size_t mimicIndex) {
  boost::property_tree::ptree pt;
  boost::property_tree::read_info(taskFile_, pt);
  std::string prefix;
  if (mimicIndex == 0) {
    prefix = "mimicJoints.left_knee.";
  } else if (mimicIndex == 1) {
    prefix = "mimicJoints.right_knee.";
  } else {
    throw std::runtime_error("No mimic joint for index: " + std::to_string(mimicIndex));
  }

  std::string parentJointName;
  std::string childJointName;
  scalar_t multiplier;  // q_child = multiplier* q_parent
  scalar_t positionGain;
  scalar_t velocityGain;

  if (verbose_) {
    std::cerr << "\n #### Joint Mimic Kinematic Constraint Config: ";
    std::cerr << "\n #### "
                 "============================================================="
                 "================\n";
  }
  loadData::loadPtreeValue(pt, parentJointName, prefix + "parentJointName", verbose_);
  loadData::loadPtreeValue(pt, childJointName, prefix + "childJointName", verbose_);
  loadData::loadPtreeValue(pt, multiplier, prefix + "multiplier", verbose_);
  loadData::loadPtreeValue(pt, positionGain, prefix + "positionGain", verbose_);
  loadData::loadPtreeValue(pt, velocityGain, prefix + "velocityGain", verbose_);
  if (verbose_) {
    std::cerr << " #### "
                 "============================================================="
                 "================\n";
  }

  JointMimicDynamicsConstraint::Config config(*mpcRobotModelPtr_, parentJointName, childJointName, multiplier, positionGain, velocityGain);

  return std::unique_ptr<StateInputConstraint>(new JointMimicDynamicsConstraint(*mpcRobotModelPtr_, config));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::unique_ptr<StateInputConstraint> WBMpcInterface::getNormalVelocityConstraint(const EndEffectorDynamics<scalar_t>& eeDynamics,
                                                                                  size_t contactPointIndex) {
  return std::unique_ptr<StateInputConstraint>(new SwingLegVerticalConstraintCppAd(*referenceManagerPtr_, eeDynamics, contactPointIndex));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

//
std::unique_ptr<StateInputCost> WBMpcInterface::getJointTorqueCost(const std::string& taskFile) {
  vector_t jointTorqueWeights(mpcRobotModelPtr_->getJointDim());
  loadData::loadEigenMatrix(taskFile, "joint_torque_weights", jointTorqueWeights);
  return std::unique_ptr<StateInputCost>(
      new JointTorqueCostCppAd(jointTorqueWeights, *pinocchioInterfacePtr_, *mpcRobotModelADPtr_, "jointTorqueCost", modelSettings_));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

// 设置 MPC 求解器
void WBMpcInterface::setupMpc() {
  // 1. 初始化求解器
  mpcPtr_.reset(new SqpMpc(mpcSettings_, sqpSettings_, *problemPtr_, *initializerPtr_));
  // 通过 mpcPtr_ 获取底层求解器指针，再调用 setReferenceManager
  mpcPtr_->getSolverPtr()->setReferenceManager(referenceManagerPtr_);
  // 2. 关键：初始化并挂载权重调整模块
  // 这样 OCS2 在 runMpc -> advanceMpc 时，会先调用该模块的 preSolverRun
  weightAdjustmentModulePtr_.reset(new MpcWeightAdjustmentModule(*this));
  mpcPtr_->getSolverPtr()->addSynchronizedModule(weightAdjustmentModulePtr_);

  // 3. 初始化 MRT
  mpcMrtPtr_.reset(new MPC_MRT_Interface(*mpcPtr_));
}

void WBMpcInterface::addSynchronizedModule(std::shared_ptr<ocs2::SolverSynchronizedModule> module) {
  if (!mpcPtr_ || !mpcPtr_->getSolverPtr()) {
    throw std::runtime_error("[WBMpcInterface::addSynchronizedModule] setupMpc() must be called first.");
  }
  mpcPtr_->getSolverPtr()->addSynchronizedModule(std::move(module));
}

// 增加一个 getter 供 python_binding 调用
std::shared_ptr<MpcWeightAdjustmentModule> WBMpcInterface::getWeightAdjustmentModule() {
  return weightAdjustmentModulePtr_;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

// 执行一步 MPC 并返回当前最优控制指令
vector_t WBMpcInterface::runMpc(const vector_t& observation, scalar_t time) {
  const size_t stateDim = mpcRobotModelPtr_->getStateDim();
  if (observation.size() != stateDim) {
    throw std::invalid_argument(
        "[WBMpcInterface::runMpc] observation size " + std::to_string(observation.size()) + " != MPC state dim " +
        std::to_string(stateDim) +
        ". Use get_state_dim() and pass OCS2 state [base_pos(3), base_euler(3), joint_pos(23), base_vel(3), euler_dot(3), joint_vel(23)].");
  }
  // 1. 设置观测值（当前仿真状态）
  SystemObservation currentObs;
  currentObs.time = time;
  currentObs.state = observation;
  currentObs.input.setZero(mpcRobotModelPtr_->getInputDim());  // 初始输入设为 0

  mpcMrtPtr_->setCurrentObservation(currentObs);

  // 2. 触发一次同步求解
  // 注：在 Python RL 模式下，通常使用同步求解（advanceMpc）来确保获取结果
  try {
    mpcMrtPtr_->advanceMpc();
  } catch (const std::exception& e) {
    return vector_t::Zero(mpcRobotModelPtr_->getInputDim());  // 异常处理
  }

  // 3. 获取最优策略在当前时刻的插值结果
  vector_t optimalInput;
  vector_t optimalState;
  size_t mode;
  mpcMrtPtr_->updatePolicy();  // 同步求解器结果
  mpcMrtPtr_->evaluatePolicy(time, observation, optimalState, optimalInput, mode);

  return optimalInput;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

// 更新 MPC 权重
void WBMpcInterface::updateMpcWeights(const ocs2::vector_t& newWeights) {
  if (weightAdjustmentModulePtr_) {
    // 2. 数据转换：将 Eigen::VectorXd (ocs2::vector_t) 转换为 std::vector<double>
    // Eigen 的 .data() 返回底层连续内存的指针
    std::vector<double> residualVec(newWeights.data(), newWeights.data() + newWeights.size());

    // 3. 调用你定义的 setResidualWeights 函数
    weightAdjustmentModulePtr_->setResidualWeights(residualVec);
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

// 设置目标状态
void WBMpcInterface::setTargetState(const vector_t& targetState) {
  // 创建一个极简的目标轨迹：在 0 时刻达到 targetState，目标输入全为 0
  TargetTrajectories targetTrajectories({0.0},                                              // 时间序列
                                        {targetState},                                      // 状态序列
                                        {vector_t::Zero(mpcRobotModelPtr_->getInputDim())}  // 输入序列
  );

  // 将目标轨迹交给参考管理器
  referenceManagerPtr_->setTargetTrajectories(targetTrajectories);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

// 设置目标轨迹
void WBMpcInterface::setTargetTrajectories(const TargetTrajectories& targetTrajectories) {
  // 将目标轨迹交给参考管理器
  referenceManagerPtr_->setTargetTrajectories(targetTrajectories);
}

}  // namespace ocs2::humanoid
