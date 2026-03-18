cmake_minimum_required(VERSION 3.8)
project(actuator)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()


# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(teacar_msgs REQUIRED)  # Your custom message package


include_directories(
  include
)

# Build executable
add_executable(pca9685_driver src/pca9685_driver.cpp)
add_executable(pwm_based_actuator src/pwm_based_actuator.cpp)

ament_target_dependencies(pca9685_driver
  rclcpp
  std_msgs
)

ament_target_dependencies(pwm_based_actuator
  rclcpp
  std_msgs
  teacar_msgs
)

# Install
install(TARGETS pca9685_driver pwm_based_actuator
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/
)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

ament_package()
