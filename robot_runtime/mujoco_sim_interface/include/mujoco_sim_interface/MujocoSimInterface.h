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

#pragma once

#include <iostream>
#include <string>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <atomic>
#include <chrono>
#include <ctime>
#include <mutex>
#include <thread>

#include <Eigen/Dense>

#include <robot_model/RobotState.h>
#include "mujoco_sim_interface/MujocoRenderer.h"
#include "mujoco_sim_interface/MujocoUtils.h"
#include "robot_core/FPSTracker.h"
#include "robot_core/Types.h"
#include "robot_model/RobotHWInterfaceBase.h"

// 添加同步模式控制宏
// 设置为 1: 启用同步模式（手动调用 step()）
// 设置为 0: 启用异步模式（后台自动运行）
#define SYNCHRONOUS_SIMULATION_MODE 1

namespace robot::mujoco_sim_interface {

struct MujocoSimConfig {
  std::string scenePath;
  std::shared_ptr<model::RobotState> initStatePtr_;
  double dt{0.0005};
  double renderFrequencyHz{60.0};
  bool headless{false};
  bool verbose{false};
};

class MujocoSimInterface : public robot::model::RobotHWInterfaceBase {
 public:
  MujocoSimInterface(const MujocoSimConfig& config, const std::string& urdfPath);

  /** Destructor */
  ~MujocoSimInterface();

  void initSim();

  void startSim();

  void simulationStep();

  // Todo Manu also reset environment
  void reset();

  /** Sync current robotStateInternal_ to MuJoCo qpos/qvel (e.g. after set_robot_state from Python). */
  void syncStateToSim();

  /** Schedule an external force pulse on a body (world frame). Applied for the next duration_steps sim steps. */
  void setPendingForce(const std::string& body_name, double fx, double fy, double fz, int duration_steps);

  /** Set sliding friction coefficient for a geom by name (e.g. "floor"). MuJoCo geom_friction[0]. */
  void setGeomFriction(const std::string& geom_name, double mu);

  // Allows the renderer to make a thread safe copy of the state at it's own frequency.
  void copyMjState(MjState& state) const;

  const mjModel* getModel() const { return mujocoModel_; }

  const MujocoSimConfig& getConfig() const { return config_; }

  // 同步模式接口
#if SYNCHRONOUS_SIMULATION_MODE
  // 直接调用单步仿真
  void step() { simulationStep(); }

  // 获取是否已初始化的状态
  bool isSimInitialized() const { return simInit_; }
#endif

 private:
  void setupJointIndexMaps();

  void setSimState(const model::RobotState& robotState);

  void updateThreadSafeRobotState();

  void simulationLoop();

  void printModelInfo();

  void updateMetrics();

  MujocoSimConfig config_;

  model::RobotState robotStateInternal_;
  mjtNum* qpos_init_;  // position                                         (nq x 1)
  mjtNum* qvel_init_;
  model::RobotJointAction robotJointActionInternal_;

  size_t timeStepMicro_;
  double simStart_;
  size_t nActiveJoints_;
  size_t nActuators_;
  std::vector<std::string> activeMuJoCoJointNames_;
  std::vector<std::string> activeMuJoCoActuatorNames_;
  std::vector<joint_index_t> activeRobotJointStateIndices_;
  std::vector<joint_index_t> activeRobotActuatorIndices_;

  mjModel* mujocoModel_ = NULL;
  mjData* mujocoData_ = NULL;
  mjContact* mujocoContact_ = NULL;
  // mjfSensor mujocoSenor_;

  bool simInit_;
  const bool headless_;
  const bool verbose_;
  std::atomic<bool> terminate_{false};
  std::atomic<bool> guiInitialized_{false};

  mutable std::mutex mujocoMutex_;  // Used to access mujoco model and data accross simulation and render threads.
#if !SYNCHRONOUS_SIMULATION_MODE
  std::thread simulate_thread_;
#endif
  std::unique_ptr<MujocoRenderer> renderer_;

  FPSTracker simFps_{"mujoco_sim"};
  std::chrono::high_resolution_clock::time_point lastRealTime_;
  Metrics metrics_{};

  size_t right_foot_sensor_addr_;
  size_t left_foot_sensor_addr_;

  size_t right_foot_touch_sensor_addr_;
  size_t left_foot_touch_sensor_addr_;

  // Optional external force pulse (for domain randomization)
  int pending_force_body_id_ = -1;
  double pending_force_[3] = {0, 0, 0};
  int pending_force_duration_ = 0;
};

}  // namespace robot::mujoco_sim_interface
