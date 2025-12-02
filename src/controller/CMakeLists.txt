cmake_minimum_required(VERSION 3.8)
project(controller)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(interfaces_msg REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(cv_bridge REQUIRED)


# Include directories
include_directories(
  include
  ${CMAKE_CURRENT_SOURCE_DIR}/../inferencer/include
)

# -------------------------------
# Joystick controller executable
add_executable(joystick_controller src/joystick_controller.cpp)
ament_target_dependencies(joystick_controller rclcpp sensor_msgs std_msgs interfaces_msg )

# Include headers
target_include_directories(joystick_controller PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)


# -------------------------------
# Param controller executable
add_executable(param_controller src/param_controller.cpp)
ament_target_dependencies(param_controller
  rclcpp std_msgs interfaces_msg)

target_include_directories(param_controller PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)


# -------------------------------
# NN controller node executable
add_executable(nn_controller_node 
  src/nn_controller_node.cpp
  ../inferencer/src/tensorflow_inferencer.cpp
)
ament_target_dependencies(nn_controller_node
  rclcpp sensor_msgs std_msgs interfaces_msg cv_bridge)

target_include_directories(nn_controller_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)



ament_export_dependencies(rosidl_default_runtime)
ament_export_dependencies(interfaces_msg)

install(TARGETS
  joystick_controller
  param_controller
  nn_controller_node
  DESTINATION lib/${PROJECT_NAME})
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)


ament_package()
