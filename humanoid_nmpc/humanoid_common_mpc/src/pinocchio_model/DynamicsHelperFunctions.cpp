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

#include <pinocchio/fwd.hpp>

#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"

// Pinnochio
#include <pinocchio/algorithm/contact-dynamics.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

#include <humanoid_common_mpc/gait/MotionPhaseDefinition.h>

namespace ocs2::humanoid {
//这是一个便捷接口，它从 pinocchioInterface 中取出模型和数据，并传递给具体的实现函数
template <typename SCALAR_T>                   //pinocchioInterface: 输入参数，包含了机器人的物理模型和当前的计算状态（modle+data）
void updateFramePlacements(const VECTOR_T<SCALAR_T>& q, PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface) {
  const auto& model = pinocchioInterface.getModel();
  auto& data = pinocchioInterface.getData();
  updateFramePlacements(q, model, data);
}
//强制编译器生成两种版本的代码：ad_ 版本（用于自动微分，如雅可比矩阵计算）；普通版本（用于常规输出）
template void updateFramePlacements(const ad_vector_t& q, PinocchioInterfaceTpl<ad_scalar_t>& pinocchioInterface);
template void updateFramePlacements(const vector_t& q, PinocchioInterfaceTpl<scalar_t>& pinocchioInterface);

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

//更新各关节坐标系位置的实现函数
template <typename SCALAR_T>
void updateFramePlacements(const VECTOR_T<SCALAR_T>& q, const pinocchio::ModelTpl<SCALAR_T>& model, pinocchio::DataTpl<SCALAR_T>& data) {
  pinocchio::forwardKinematics(model, data, q);  //pinocchio::forwardKinematics函数：正运动学计算，根据关节角度q，计算每一个关节相对于世界坐标系的位置和姿态
  updateFramePlacements(model, data);
}
template void updateFramePlacements(const ad_vector_t& q,
                                    const pinocchio::ModelTpl<ad_scalar_t>& model,
                                    pinocchio::DataTpl<ad_scalar_t>& data);
template void updateFramePlacements(const vector_t& q, const pinocchio::ModelTpl<scalar_t>& model, pinocchio::DataTpl<scalar_t>& data);

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

//计算接触点位置：给定关节状态q，返回所有预设接触点（通常是双足机器人的左右脚底）的3D世界坐标
template <typename SCALAR_T>
std::vector<VECTOR3_T<SCALAR_T>> computeContactPositions(const VECTOR_T<SCALAR_T>& q,
                                                         PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface,
                                                         const MpcRobotModelBase<SCALAR_T>& mpcRobotModel) {
  updateFramePlacements<SCALAR_T>(q, pinocchioInterface);
  return getContactPositions<SCALAR_T>(pinocchioInterface, mpcRobotModel);
}
template std::vector<VECTOR3_T<ad_scalar_t>> computeContactPositions(const VECTOR_T<ad_scalar_t>& q,
                                                                     PinocchioInterfaceTpl<ad_scalar_t>& pinocchioInterface,
                                                                     const MpcRobotModelBase<ad_scalar_t>& mpcRobotModel);
template std::vector<VECTOR3_T<scalar_t>> computeContactPositions(const VECTOR_T<scalar_t>& q,
                                                                  PinocchioInterfaceTpl<scalar_t>& pinocchioInterface,
                                                                  const MpcRobotModelBase<scalar_t>& mpcRobotModel);

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

//提取所有接触点（如足端）在世界坐标系下的位置坐标
template <typename SCALAR_T>                       //mpcRobotModel: 包含了机器人特定的配置信息
std::vector<VECTOR3_T<SCALAR_T>> getContactPositions(const PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface,
                                                     const MpcRobotModelBase<SCALAR_T>& mpcRobotModel) {
  assert(mpcRobotModel.modelSettings.contactNames.size() == N_CONTACTS);
  std::vector<VECTOR3_T<SCALAR_T>> footPositions;   //为结果列表预分配内存
  footPositions.reserve(N_CONTACTS);
  const auto& data = pinocchioInterface.getData();
  std::vector<pinocchio::FrameIndex> contactFrameIndices = getContactFrameIndices(pinocchioInterface, mpcRobotModel);

  for (size_t i = 0; i < N_CONTACTS; i++) {
    const VECTOR3_T<SCALAR_T>& footPosition = data.oMf[getContactFrameIndex(pinocchioInterface, mpcRobotModel, i)].translation();
    //data.oMf[index]: 这是 Pinocchio 的核心数据结构，代表“世界坐标系下（o）该框架（f）的变换矩阵（M）”
    //.translation(): 从 4x4 的变换矩阵中提取出前 3 行第 4 列的平移向量，即该接触点在空间中的 X, Y, Z 坐标
    footPositions.emplace_back(footPosition);
  }
  return footPositions;
}
template std::vector<VECTOR3_T<ad_scalar_t>> getContactPositions(const PinocchioInterfaceTpl<ad_scalar_t>& pinocchioInterface,
                                                                 const MpcRobotModelBase<ad_scalar_t>& mpcRobotModel);
template std::vector<VECTOR3_T<scalar_t>> getContactPositions(const PinocchioInterfaceTpl<scalar_t>& pinocchioInterface,
                                                              const MpcRobotModelBase<scalar_t>& mpcRobotModel);

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

//frameNames: 你想要查询位置的框架名称列表（如 {"hand_l", "hand_r"}
template <typename SCALAR_T>
std::vector<VECTOR3_T<SCALAR_T>> computeFramePositions(const VECTOR_T<SCALAR_T>& q,
                                                       PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface,
                                                       std::vector<std::string> frameNames) {
  updateFramePlacements<SCALAR_T>(q, pinocchioInterface);
  return getFramePositions<SCALAR_T>(pinocchioInterface, frameNames);
}
template std::vector<VECTOR3_T<ad_scalar_t>> computeFramePositions(const VECTOR_T<ad_scalar_t>& q,
                                                                   PinocchioInterfaceTpl<ad_scalar_t>& pinocchioInterface,
                                                                   std::vector<std::string> frameNames);
template std::vector<VECTOR3_T<scalar_t>> computeFramePositions(const VECTOR_T<scalar_t>& q,
                                                                PinocchioInterfaceTpl<scalar_t>& pinocchioInterface,
                                                                std::vector<std::string> frameNames);

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

template <typename SCALAR_T>
std::vector<VECTOR3_T<SCALAR_T>> getFramePositions(const PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface,
                                                   std::vector<std::string> frameNames) {
  std::vector<VECTOR3_T<SCALAR_T>> positions;
  positions.reserve(frameNames.size());
  const auto& data = pinocchioInterface.getData();
  for (size_t i = 0; i < frameNames.size(); i++) {
    const pinocchio::FrameIndex frameIndex = pinocchioInterface.getModel().getFrameId(frameNames[i]);
    const VECTOR3_T<SCALAR_T>& position = data.oMf[frameIndex].translation();
    positions.emplace_back(position);
  }
  return positions;
}
template std::vector<VECTOR3_T<ad_scalar_t>> getFramePositions(const PinocchioInterfaceTpl<ad_scalar_t>& pinocchioInterface,
                                                               std::vector<std::string> frameNames);
template std::vector<VECTOR3_T<scalar_t>> getFramePositions(const PinocchioInterfaceTpl<scalar_t>& pinocchioInterface,
                                                            std::vector<std::string> frameNames);

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

//实时估算机器人脚下的地面的高度
//参数 measuredMode ：当前的步态模式编号
scalar_t computeGroundHeightEstimate(PinocchioInterfaceTpl<scalar_t>& pinocchioInterface,
                                     const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                                     const vector_t& q,
                                     size_t measuredMode) {
  updateFramePlacements<scalar_t>(q, pinocchioInterface);
  return getGroundHeightEstimate(pinocchioInterface, mpcRobotModel, measuredMode);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

scalar_t getGroundHeightEstimate(PinocchioInterfaceTpl<scalar_t>& pinocchioInterface,
                                 const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                                 size_t measuredMode) {
  contact_flag_t measuredContactFlags = modeNumber2StanceLeg(measuredMode); //将步态模式编号measuredMode 转换为“接触标志位”（如[true, false]

  std::vector<vector3_t> contactPositions = getContactPositions<scalar_t>(pinocchioInterface, mpcRobotModel);

  static scalar_t terrainHeight = 0.0;

  // Use right foot if in contact 双脚支撑取z舟坐标平均值；单脚支撑取单脚z轴值
  if (measuredContactFlags[0] && measuredContactFlags[1]) {
    vector3_t footPosition1 = contactPositions[0];
    vector3_t footPosition2 = contactPositions[1];
    terrainHeight = 0.5 * (footPosition1[2] + footPosition2[2]);
  } else if (measuredContactFlags[0]) {
    vector3_t footPosition = contactPositions[0];
    terrainHeight = footPosition[2];
  } else if (measuredContactFlags[1]) {
    vector3_t footPosition = contactPositions[1];
    terrainHeight = footPosition[2];
  }
  return terrainHeight;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

//M：全系统的广义质量矩阵（Inertia Matrix）。
//nle：非线性效应项（包含科里奥利力、离心力、重力项）。
//qdd_joints：已知的关节加速度（通常是控制输入 u 的一部分）。
//externalForcesInJointSpace：作用在关节空间中的外力项（通常是足端接触力的映射）。
//代码实现的其实是动力学方程 Mq¨​+nle=τ+external_forces 的基座部分
template <typename SCALAR_T>
VECTOR6_T<SCALAR_T> computeBaseAcceleration(const MATRIX_T<SCALAR_T>& M,
                                            const VECTOR_T<SCALAR_T>& nle,
                                            const VECTOR_T<SCALAR_T>& qdd_joints,
                                            const VECTOR_T<SCALAR_T>& externalForcesInJointSpace) {
  // Due to the block diagonal structure of the generalized mass matrix corresponding to the base the base mass matrix can be split into a
  // linear and angular part. Which are both inverted separately. This does not only exploit part of the sparsity but also prevents a CppAD
  // branching error when multiplying a 6x6 matrix with a6 dim. vector.

  Eigen::Matrix<SCALAR_T, 3, 3> M_bb_lin = M.topLeftCorner(3, 3);  //提取基座质量矩阵的左上角 3x3 块，对应线性运动惯性
  Eigen::Matrix<SCALAR_T, 3, 3> M_bb_ang = M.block(3, 3, 3, 3);    //提取接下来的对角线 3x3 块，对应转动惯性
  auto M_bj = M.block(0, 6, 6, qdd_joints.size());
  Eigen::Matrix<SCALAR_T, 3, 3> M_bb_lin_inv = M_bb_lin.inverse();
  Eigen::Matrix<SCALAR_T, 3, 3> M_bb_ang_inv = M_bb_ang.inverse();
  //intermediate：计算作用在基座上的“净力/力矩”
  VECTOR6_T<SCALAR_T> intermediate = -nle.head(6) - M_bj * qdd_joints + externalForcesInJointSpace.head(6);

  VECTOR6_T<SCALAR_T> baseAccelerations;
  baseAccelerations.head(3) = M_bb_lin_inv * intermediate.head(3); //基座的线性加速度
  baseAccelerations.tail(3) = M_bb_ang_inv * intermediate.tail(3); //基座的角加速度

  return baseAccelerations;
}
template VECTOR6_T<scalar_t> computeBaseAcceleration(const MATRIX_T<scalar_t>& M,
                                                     const VECTOR_T<scalar_t>& nle,
                                                     const VECTOR_T<scalar_t>& qdd_joints,
                                                     const VECTOR_T<scalar_t>& externalForcesInJointSpace);
template VECTOR6_T<ad_scalar_t> computeBaseAcceleration(const MATRIX_T<ad_scalar_t>& M,
                                                        const VECTOR_T<ad_scalar_t>& nle,
                                                        const VECTOR_T<ad_scalar_t>& qdd_joints,
                                                        const VECTOR_T<ad_scalar_t>& externalForcesInJointSpace);

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

// 逆动力学计算，根据机器人当前的状态、期望的加速度以及受到的外力，反算出关节电机需要输出的扭矩
template <typename SCALAR_T>
VECTOR_T<SCALAR_T> computeJointTorques(const VECTOR_T<SCALAR_T>& q,  //机器人的广义位置
                                       const VECTOR_T<SCALAR_T>& qd,  //机器人的广义速度
                                       const VECTOR_T<SCALAR_T>& qdd_joints,
                                       const std::array<VECTOR6_T<SCALAR_T>, 2>& footWrenches,  //左右脚部受到的 6 维外力
                                       PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface) {
  const auto& model = pinocchioInterface.getModel();
  pinocchio::DataTpl<SCALAR_T>& data = pinocchioInterface.getData();

  pinocchio::crba(model, data, q);  //计算全系统的广义质量矩阵 M ，并存入 data.M
  pinocchio::nonLinearEffects(model, data, q, qd);  //计算非线性力项项（包含重力、离心力、科氏力） nle ，存入 data.nle

  // Compute Jacobians for the foot frames // 获取左、右脚接触点在当前位姿下的 6xN 雅可比矩阵
  MATRIX_T<SCALAR_T> J_foot_l = MATRIX_T<SCALAR_T>::Zero(6, qd.size());
  MATRIX_T<SCALAR_T> J_foot_r = MATRIX_T<SCALAR_T>::Zero(6, qd.size());

  ////////////////////////////////////////////////////////////////////////////

  pinocchio::computeFrameJacobian(model, data, q, model.getFrameId("foot_l_contact"), pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
                                  J_foot_l);
  pinocchio::computeFrameJacobian(model, data, q, model.getFrameId("foot_r_contact"), pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
                                  J_foot_r);

  // Project contact wrenches into the joint space 通过雅可比矩阵的转置，将足端受到的地面反作用力转化为作用在所有广义坐标上的等效力

  VECTOR_T<SCALAR_T> externalForcesInJointSpace = J_foot_l.transpose() * footWrenches[0] + J_foot_r.transpose() * footWrenches[1];

  VECTOR6_T<SCALAR_T> baseAccelerations = computeBaseAcceleration(data.M, data.nle, qdd_joints, externalForcesInJointSpace);

  VECTOR_T<SCALAR_T> q_dd(qd.size());
  q_dd << baseAccelerations, qdd_joints;  //将解算出的基座加速度和给定的关节加速度拼成完整的广义加速度向量
  size_t n_joints = qdd_joints.size();

  //动力学公式：实现方程 τ=Mq¨​+C(q,q˙​)−τext​
  VECTOR_T<SCALAR_T> jointTorques =
      data.M.bottomRows(n_joints) * q_dd + data.nle.tail(n_joints) - externalForcesInJointSpace.tail(n_joints);

  // return jointTorques;
  return jointTorques;
}
template VECTOR_T<scalar_t> computeJointTorques(const VECTOR_T<scalar_t>& q,
                                                const VECTOR_T<scalar_t>& qd,
                                                const VECTOR_T<scalar_t>& qdd_joints,
                                                const std::array<VECTOR6_T<scalar_t>, 2>& footWrenches,
                                                PinocchioInterfaceTpl<scalar_t>& pinocchioInterface);
template VECTOR_T<ad_scalar_t> computeJointTorques(const VECTOR_T<ad_scalar_t>& q,
                                                   const VECTOR_T<ad_scalar_t>& qd,
                                                   const VECTOR_T<ad_scalar_t>& qdd_joints,
                                                   const std::array<VECTOR6_T<ad_scalar_t>, 2>& footWrenches,
                                                   PinocchioInterfaceTpl<ad_scalar_t>& pinocchioInterface);

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

//使用递归牛顿-欧拉算法 (Recursive Newton-Euler Algorithm, RNEA)计算关节力矩，处理复杂多体系统时通常具有更高的计算效率
template <typename SCALAR_T>
VECTOR_T<SCALAR_T> computeJointTorquesRNEA(const VECTOR_T<SCALAR_T>& q,
                                           const VECTOR_T<SCALAR_T>& qd,
                                           const VECTOR_T<SCALAR_T>& qdd_joints,
                                           const std::array<VECTOR6_T<SCALAR_T>, 2>& footWrenches,
                                           PinocchioInterfaceTpl<SCALAR_T>& pinocchioInterface) {
  const auto& model = pinocchioInterface.getModel();
  auto& data = pinocchioInterface.getData();

  pinocchio::container::aligned_vector<pinocchio::Force> fextDesired(model.njoints, pinocchio::Force::Zero()); //创建一个向量存储作用在每个关节上的外部力

  pinocchio::forwardKinematics(model, data, q, qd);
  pinocchio::updateFramePlacements(model, data);

  // RNEA 算法需要知道作用在每个连杆上的外部力（在连杆本地坐标系下），由足端力计算得到
  auto setExternalForce = [&](const std::string& frameName, size_t i) {
    const auto frameIndex = model.getFrameId(frameName);
    const auto jointIndex = model.frames[frameIndex].parentJoint;
    const VECTOR3_T<SCALAR_T> translationJointFrameToContactFrame = model.frames[frameIndex].placement.translation();
    const MATRIX3_T<SCALAR_T> rotationWorldFrameToJointFrame = data.oMi[jointIndex].rotation().transpose(); //世界坐标系到关节坐标系的旋转矩阵
    const VECTOR3_T<SCALAR_T> contactForce = rotationWorldFrameToJointFrame * footWrenches[i].head(3);
    const VECTOR3_T<SCALAR_T> contactTorque = rotationWorldFrameToJointFrame * footWrenches[i].tail(3);
    fextDesired[jointIndex].linear() = contactForce;  //直接赋值旋转后的力
    fextDesired[jointIndex].angular() = translationJointFrameToContactFrame.cross(contactForce) + contactTorque;  //利用叉乘公式 τ=r×f 计算由于力作用点偏移产生的力矩，并叠加输入自带的力矩
  };

  setExternalForce("foot_l_contact", 0);
  setExternalForce("foot_r_contact", 1);

  pinocchio::crba(model, data, q);
  pinocchio::nonLinearEffects(model, data, q, qd);

  // Compute Jacobians for the foot frames
  MATRIX_T<SCALAR_T> J_foot_l = MATRIX_T<SCALAR_T>::Zero(6, qd.size());
  MATRIX_T<SCALAR_T> J_foot_r = MATRIX_T<SCALAR_T>::Zero(6, qd.size());

  ////////////////////////////////////////////////////////////////////////////

  pinocchio::computeFrameJacobian(model, data, q, model.getFrameId("foot_l_contact"), pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
                                  J_foot_l);
  pinocchio::computeFrameJacobian(model, data, q, model.getFrameId("foot_r_contact"), pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
                                  J_foot_r);

  // Project contact wrenches into the joint space

  VECTOR_T<SCALAR_T> externalForcesInJointSpace = J_foot_l.transpose() * footWrenches[0] + J_foot_r.transpose() * footWrenches[1];

  // Repalce q with external forces in joint space.

  VECTOR6_T<SCALAR_T> baseAccelerations = computeBaseAcceleration(data.M, data.nle, qdd_joints, externalForcesInJointSpace);

  VECTOR_T<SCALAR_T> q_dd(qd.size());
  q_dd << baseAccelerations, qdd_joints;

  vector_t torques = pinocchio::rnea(model, data, q, qd, q_dd, fextDesired);

  return torques.tail(qdd_joints.size());
}
template VECTOR_T<scalar_t> computeJointTorquesRNEA(const VECTOR_T<scalar_t>& q,
                                                    const VECTOR_T<scalar_t>& qd,
                                                    const VECTOR_T<scalar_t>& qdd_joints,
                                                    const std::array<VECTOR6_T<scalar_t>, 2>& footWrenches,
                                                    PinocchioInterfaceTpl<scalar_t>& pinocchioInterface);
// template VECTOR_T<ad_scalar_t> computeJointTorquesRNEA(const VECTOR_T<ad_scalar_t>& q,
//                                                        const VECTOR_T<ad_scalar_t>& qd,
//                                                        const VECTOR_T<ad_scalar_t>& qdd_joints,
//                                                        const std::array<VECTOR6_T<ad_scalar_t>, 2>& footWrenches,
//                                                        PinocchioInterfaceTpl<ad_scalar_t>& pinocchioInterface);

}  // namespace ocs2::humanoid
