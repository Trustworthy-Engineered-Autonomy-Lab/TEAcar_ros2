cmake_minimum_required(VERSION 3.8)
project(nn_controller)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(donkeycar_msgs REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---- Mock plugin ----
add_library(tensorflow_inferencer SHARED plugins/tensorflow_inferencer.cpp)
target_include_directories(tensorflow_inferencer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---- Node library ----
add_library(nn_controller_lib SHARED
  src/nn_controller_node.cpp
)
ament_target_dependencies(nn_controller_lib
  rclcpp sensor_msgs std_msgs cv_bridge donkeycar_msgs
)
target_include_directories(nn_controller_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(nn_controller_lib ${OpenCV_LIBS})

# ---- Executable ----
add_executable(nn_controller_node
  src/nn_controller_main.cpp
)
target_link_libraries(nn_controller_node
  nn_controller_lib
  tensorflow_inferencer
)
ament_target_dependencies(nn_controller_node rclcpp)

# ---- Install ----
install(TARGETS
  nn_controller_lib
  nn_controller_node
  tensorflow_inferencer
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/ DESTINATION include)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/launch")
  install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/config")
  install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)
endif()

ament_export_include_directories(include)
ament_export_dependencies(rclcpp sensor_msgs std_msgs cv_bridge OpenCV donkeycar_msgs)
ament_package()
