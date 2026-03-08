cmake_minimum_required(VERSION 3.5)
project(humanoid_wb_mpc)
# 强制要求所有目标都生成位置无关代码
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Werror=return-type)
endif ()

# Generate compile_commands.json for clang tools
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 1. 先找到 Python 开发库,再查找 pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
# 显式查找 Abseil 基础库
find_package(absl REQUIRED)

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(ocs2_core REQUIRED)   # 补齐缺失的依赖
find_package(ocs2_oc REQUIRED)
find_package(ocs2_mpc REQUIRED)
find_package(ocs2_ddp REQUIRED)
find_package(ocs2_sqp REQUIRED)
find_package(ocs2_ros2_interfaces REQUIRED)
find_package(ocs2_pinocchio_interface REQUIRED)
find_package(ocs2_centroidal_model REQUIRED)
find_package(humanoid_common_mpc REQUIRED)
find_package(robot_model REQUIRED)
find_package(mujoco_vendor REQUIRED)
# mujoco_vendor 通过 ament 导出时可能不传递 INTERFACE 库路径，此处用安装路径显式指定 libmujoco
# mujoco_vendor_DIR = <install_base>/mujoco_vendor/share/mujoco_vendor/cmake -> 向上 3 级得到包前缀
get_filename_component(_mujoco_vendor_cmake "${mujoco_vendor_DIR}" PATH)
get_filename_component(_mujoco_vendor_share "${_mujoco_vendor_cmake}" PATH)
get_filename_component(_mujoco_vendor_prefix "${_mujoco_vendor_share}" PATH)
set(MUJOCO_LIBRARY "${_mujoco_vendor_prefix}/lib/libmujoco.so")
if(NOT EXISTS "${MUJOCO_LIBRARY}")
  set(MUJOCO_LIBRARY "${_mujoco_vendor_prefix}/lib/libmujoco.so.3.3.4")
endif()
find_package(mujoco_sim_interface REQUIRED)

find_package(GTest REQUIRED)

find_package(Boost REQUIRED COMPONENTS
  system
  filesystem
  log
  log_setup
)

# find_package(PkgConfig REQUIRED)
pkg_check_modules(pinocchio REQUIRED pinocchio)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

###########
## Build ##
###########

set(CMAKE_CXX_STANDARD 20)

set(FLAGS ${OCS2_CXX_FLAGS} -Wno-invalid-partial-specialization)

# 计算项目根目录路径 (修正：向上两级以定位到 wb_humanoid_mpc)
get_filename_component(PROJECT_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

include_directories(
  include
  test
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${pinocchio_INCLUDE_DIRS}
  ${GTEST_INCLUDE_DIRS}
  # 添加Mujoco仿真接口头文件路径
  ${PROJECT_ROOT_DIR}/robot_runtime/mujoco_sim_interface/include
  ${PROJECT_ROOT_DIR}/robot_runtime/robot_model/include
  ${PROJECT_ROOT_DIR}/robot_runtime/robot_core/include
)

set(dependencies
  ament_cmake
  ament_index_cpp
  rclcpp
  ocs2_core
  ocs2_centroidal_model
  ocs2_ros2_interfaces
  ocs2_mpc
  ocs2_sqp
  ocs2_pinocchio_interface
  humanoid_common_mpc
  robot_model
  mujoco_vendor
  ocs2_ddp   #加入dependencies列表,方便后续导出
)

link_directories(
  ${pinocchio_LIBRARY_DIRS}
)

add_definitions("-DBOOST_MPL_LIMIT_LIST_SIZE=30")

#############
## Library ##
#############

add_library(${PROJECT_NAME}
  src/dynamics/DynamicsHelperFunctions.cpp
  src/dynamics/WBAccelDynamicsAD.cpp
  src/end_effector/PinocchioEndEffectorDynamicsCppAd.cpp
  src/constraint/EndEffectorDynamicsLinearAccConstraint.cpp
  src/constraint/EndEffectorDynamicsAccelerationsConstraint.cpp
  src/constraint/ZeroAccelerationConstraintCppAd.cpp
  src/constraint/SwingLegVerticalConstraintCppAd.cpp
  src/constraint/JointMimicDynamicsConstraint.cpp
  src/WBMpcInterface.cpp
  src/WBMpcPreComputation.cpp
  src/cost/EndEffectorDynamicsCostHelpers.cpp
  src/cost/EndEffectorDynamicsQuadraticCost.cpp
  src/cost/EndEffectorDynamicsFootCost.cpp
  src/cost/JointTorqueCostCppAd.cpp
  src/command/WBMpcTargetTrajectoriesCalculator.cpp
  src/mrt/WBMpcMrtJointController.cpp
  src/synchronized_module/MpcWeightAdjustmentModule.cpp # 新增
)
ament_target_dependencies(${PROJECT_NAME}
  ocs2_core
  ocs2_oc
  ocs2_mpc
  ocs2_ddp
  ocs2_sqp
  ocs2_ros2_interfaces
  ocs2_pinocchio_interface
  ocs2_centroidal_model
  humanoid_common_mpc
  robot_model
  rclcpp
  mujoco_sim_interface
)

target_link_libraries(${PROJECT_NAME}
  absl::raw_hash_set
  absl::hash
  absl::base
  absl::strings
  absl::raw_logging_internal
)

## 2. 创建 Python 模块 ##
pybind11_add_module(humanoid_wb_mpc_py
  src/python/python_binding.cpp
)

target_include_directories(humanoid_wb_mpc_py PRIVATE
  ${PROJECT_ROOT_DIR}/robot_runtime/mujoco_sim_interface/include
  ${PROJECT_ROOT_DIR}/robot_runtime/robot_model/include
  ${PROJECT_ROOT_DIR}/robot_runtime/robot_core/include
)

# 链接所有库（使用 PRIVATE 关键字以与 pybind11 保持兼容）
target_link_libraries(humanoid_wb_mpc_py PRIVATE
    ${PROJECT_NAME}             # 链接你自己的 C++ 库
    pybind11::module            # pybind11 必须
    Python3::Python
    ${MUJOCO_LIBRARY}           # mujoco_vendor 安装的 libmujoco.so（显式路径避免 -lmujoco 找不到）
    glfw
    GLEW
    ${Boost_LIBRARIES}
    absl::flat_hash_map
    absl::raw_hash_set
    absl::base
    absl::strings
    absl::hash
)

target_compile_options(${PROJECT_NAME} PUBLIC ${OCS2_CXX_FLAGS})


#############
## Install ##
#############

install(TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

# 将头文件物理安装到 install 目录
install(
  DIRECTORY include/
  DESTINATION include
)

# 将模块安装到 lib/pythonX.Y/site-packages，与 colcon 自动加入的 PYTHONPATH 一致，避免依赖 environment hook
install(TARGETS humanoid_wb_mpc_py
  LIBRARY DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages
  RUNTIME DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages
)

# install(TARGETS
#   test_pinocchio_model
#   test_humanoid_interface
#   test_contact_moment_xy_constraint
#   test_end_effector_kinematic_cost_element
#   DESTINATION lib/${PROJECT_NAME}
# )

# install(DIRECTORY launch rviz
#   DESTINATION share/${PROJECT_NAME}
# )

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)  # 导出目标
ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_dependencies(${dependencies})

# 安装 ament hook 文件用于自动设置 PYTHONPATH
install(
  DIRECTORY share/${PROJECT_NAME}/hook
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY share/${PROJECT_NAME}/environment
  DESTINATION share/${PROJECT_NAME}
)

ament_package()



