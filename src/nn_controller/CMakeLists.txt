cmake_minimum_required(VERSION 3.8)
project(nn_controller)

find_package(ament_cmake REQUIRED)

# Install configuration files (nn_controller.yaml, etc.)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/config")
  install(DIRECTORY config/
    DESTINATION share/${PROJECT_NAME}/config)
endif()

ament_package()
