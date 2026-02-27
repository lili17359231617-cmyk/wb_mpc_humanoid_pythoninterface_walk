#include "humanoid_wb_mpc/synchronized_module/MpcWeightAdjustmentModule.h"
#include <humanoid_common_mpc/cost/StateInputQuadraticCost.h>
#include "humanoid_wb_mpc/WBMpcInterface.h"  // 确保能看到 Interface 的完整定义

namespace ocs2::humanoid {

MpcWeightAdjustmentModule::MpcWeightAdjustmentModule(WBMpcInterface& interface) : interface_(interface) {
  currentResidual_.assign(70, 0.0);  // 初始残差为0
}

// 这个函数供外部（将来是 Python）高频调用
void MpcWeightAdjustmentModule::setResidualWeights(const std::vector<double>& residualActions) {
  if (residualActions.size() != 70) return;

  std::lock_guard<std::mutex> lock(residualMutex_);
  currentResidual_ = residualActions;
  hasNewResidual_ = true;
}

// OCS2 求解器调用的钩子
void MpcWeightAdjustmentModule::preSolverRun(scalar_t initTime,
                                             scalar_t finalTime,
                                             const vector_t& initState,
                                             const ReferenceManagerInterface& referenceManager) {
  // 只有当 RL 下发了新权重时，才执行耗时的矩阵运算
  if (hasNewResidual_) {
    updateWeightsInternal();
    hasNewResidual_ = false;  // 重置标志位
  }
}

// 核心逻辑实现
void MpcWeightAdjustmentModule::updateWeightsInternal() {
  auto& ocp = interface_.getOptimalControlProblem();

  try {
    // 1. 获取代价项。因为 Q_ 现在在基类里是 public，我们转成基类即可
    auto& costTerm = ocp.costPtr->get<QuadraticStateInputCost>("stateInputQuadraticCost");

    std::lock_guard<std::mutex> lock(residualMutex_);

    // 2. 现在可以合法地访问 costTerm.Q_ 了
    matrix_t Q = costTerm.Q_;

    if (static_cast<Eigen::Index>(currentResidual_.size()) == Q.rows()) {
      for (int i = 0; i < Q.rows(); ++i) {
        // 执行残差法映射：Q_i = Q_base_i * exp(a_i)
        Q(i, i) *= std::exp(currentResidual_[i]);
      }
      // 3. 写回修改后的矩阵
      costTerm.Q_ = Q;
    }
  } catch (const std::exception& e) {
    std::cerr << "[MpcWeightAdjustmentModule] Error: " << e.what() << std::endl;
  }
}

}  // namespace ocs2::humanoid