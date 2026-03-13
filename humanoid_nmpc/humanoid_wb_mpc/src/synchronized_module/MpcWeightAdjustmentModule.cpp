#include "humanoid_wb_mpc/synchronized_module/MpcWeightAdjustmentModule.h"
#include <humanoid_common_mpc/cost/StateInputQuadraticCost.h>
#include "humanoid_wb_mpc/WBMpcInterface.h"  // 确保能看到 Interface 的完整定义
#include <cmath>

namespace ocs2::humanoid {

MpcWeightAdjustmentModule::MpcWeightAdjustmentModule(WBMpcInterface& interface) : interface_(interface) {
  currentResidual_.assign(58, 0.0);  // 初始残差为0，与 Q 矩阵状态维一致
}

// 这个函数供外部（将来是 Python）高频调用
void MpcWeightAdjustmentModule::setResidualWeights(const std::vector<double>& residualActions) {
  if (residualActions.size() != 58) return;

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

    // 2. 首次调用时从 OCP 快照保存基准 Q，后续每次均从基准出发，
    //    避免 exp(a) 在多步间累乘导致权重无限增长/衰减。
    if (Q_base_.rows() == 0) {
      Q_base_ = costTerm.Q_;
    }

    if (static_cast<Eigen::Index>(currentResidual_.size()) == Q_base_.rows()) {
      constexpr scalar_t kZeroWeightEps = 1e-9;
      constexpr scalar_t kZeroWeightAlpha = 1.0;  // Softplus 映射的缩放系数 α
      constexpr scalar_t kZeroWeightBias = 0.0;   // Softplus 映射的偏置 b

      matrix_t Q = Q_base_;
      for (int i = 0; i < Q.rows(); ++i) {
        const scalar_t qBase = Q_base_(i, i);
        const scalar_t a = static_cast<scalar_t>(currentResidual_[i]);

        if (std::fabs(qBase) < kZeroWeightEps) {
          // 初始权重为 0：使用 Softplus 映射
          // Q_new = α * ln(1 + exp(a + b))
          Q(i, i) = kZeroWeightAlpha * std::log1p(std::exp(a + kZeroWeightBias));
        } else {
          // 初始权重非 0：采用指数残差
          // Q_new = Q_base * exp(a)
          Q(i, i) = qBase * std::exp(a);
        }
      }
      // 3. 写回修改后的矩阵
      costTerm.Q_ = Q;
    }
  } catch (const std::exception& e) {
    std::cerr << "[MpcWeightAdjustmentModule] Error: " << e.what() << std::endl;
  }
}

}  // namespace ocs2::humanoid