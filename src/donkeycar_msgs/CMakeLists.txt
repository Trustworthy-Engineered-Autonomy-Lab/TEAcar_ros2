cmake_minimum_required(VERSION 3.8)
project(donkeycar_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
#find_package(std_msgs REQUIRED)

#set(
#)

#rosidl_generate_interfaces(${PROJECT_NAME}
#)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
