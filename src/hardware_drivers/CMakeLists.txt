cmake_minimum_required(VERSION 3.5)
project(hardware_drivers)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

include_directories(
  include
)

add_library(pca9685 src/pca9685.cpp)

target_link_libraries(pca9685
  i2c
)

add_executable(pca9685_driver src/pca9685_driver.cpp)

ament_target_dependencies(pca9685_driver
  rclcpp
  std_msgs
)

target_link_libraries(pca9685_driver
  pca9685
)

add_library(mpu6050 src/mpu6050.cpp)
target_link_libraries(mpu6050 i2c)

add_executable(mpu6050_driver src/mpu6050_driver.cpp)

ament_target_dependencies(mpu6050_driver
  rclcpp
  std_msgs
  sensor_msgs
)

target_link_libraries(mpu6050_driver
  mpu6050
)

install(
  TARGETS pca9685_driver mpu6050 mpu6050_driver
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
