#pragma once

#include <ocs2_core/cost/QuadraticStateInputCost.h>
#include <ocs2_oc/synchronized_module/SolverSynchronizedModule.h>
#include <atomic>
#include <mutex>
#include <vector>

namespace ocs2::humanoid {

class WBMpcInterface;  // 添加前向声明： WBMpcInterface 类

/**
 * @brief 同步模块：在每一轮 MPC 求解前更新状态代价矩阵 Q
 */
class MpcWeightAdjustmentModule : public SolverSynchronizedModule {
 public:
  // 构造函数：需要引用 interface 以便后续访问 OCP
  explicit MpcWeightAdjustmentModule(WBMpcInterface& interface);

  // OCS2 核心钩子：在 Solver 运行前同步调用
  void preSolverRun(scalar_t initTime,
                    scalar_t finalTime,
                    const vector_t& initState,
                    const ReferenceManagerInterface& referenceManager) override;

  // Solver 运行后的逻辑（本需求暂不需要，保留空实现）
  void postSolverRun(const PrimalSolution& primalSolution) override {}

  /**
   * @brief 供外部调用（如 Python 封装层）设置最新的残差动作
   * @param residualActions 58 维向量，对应 RL 策略的输出 a（与 MPC 状态/Q 维一致）
   */
  void setResidualWeights(const std::vector<double>& residualActions);

  /**
   * @brief 获取当前一次 MPC 求解前使用的 Q 矩阵对角线（Q_new 对角元素）
   *
   * 注意：仅对已经调用过一次 updateWeightsInternal() 之后有效。
   */
  std::vector<double> getCurrentQDiag();

  /**
   * @brief 获取上一轮 MPC 求解前使用的 Q 矩阵对角线
   *
   * 若尚未有「上一轮」，则通常与当前对角线相同。
   */
  std::vector<double> getPrevQDiag();

 private:
  void updateWeightsInternal();
  WBMpcInterface& interface_;

  std::vector<double> currentResidual_;
  matrix_t Q_base_;                          // 首次调用时从 OCP 快照保存的基准 Q 矩阵
  vector_t currentQDiag_;                    // 当前 Q 矩阵的对角线 (Q_new 对角元素)
  vector_t prevQDiag_;                       // 上一轮 Q 矩阵的对角线
  std::mutex residualMutex_;                 // 保证多线程安全（同时保护残差与 Q 对角线）
  std::atomic<bool> hasNewResidual_{false};  // 标记是否有待更新的数据
};

}  // namespace ocs2::humanoid