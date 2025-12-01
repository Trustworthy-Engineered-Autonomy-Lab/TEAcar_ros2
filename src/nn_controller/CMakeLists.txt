cmake_minimum_required(VERSION 3.8)
project(nn_controller)

find_package(ament_cmake REQUIRED)

# Install config files so other packages/launch files can load them
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
