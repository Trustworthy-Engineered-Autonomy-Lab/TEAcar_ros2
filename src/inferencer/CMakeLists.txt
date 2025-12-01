cmake_minimum_required(VERSION 3.5)
project(inferencer)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

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

# TensorFlow inferencer library
add_library(tensorflow_inferencer SHARED
  src/tensorflow_inferencer.cpp
)

target_include_directories(tensorflow_inferencer
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

install(
  TARGETS tensorflow_inferencer
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

ament_export_libraries(tensorflow_inferencer)

# TensorRT inferencer library
if(EXISTS "/usr/local/cuda/include/cuda_runtime_api.h")
  message(STATUS "Building TensorRT inferencer with CUDA headers")

  add_library(tensorrt_inferencer SHARED
    src/tensorrt_inferencer.cpp
  )

  target_include_directories(tensorrt_inferencer
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
    PRIVATE
      /usr/local/cuda/include
  )

  install(
    TARGETS tensorrt_inferencer
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
  )
else()
  message(WARNING "CUDA not found; tensorrt_inferencer will NOT be built on this machine")
endif()


# Install public headers
install(DIRECTORY include/ DESTINATION include)

# Let downstream packages include our headers (e.g. #include "inferencer/inferencer_api.hpp")
ament_export_include_directories(include)

ament_package()
