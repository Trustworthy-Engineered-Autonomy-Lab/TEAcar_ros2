cmake_minimum_required(VERSION 3.5)
project(inferencer)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++17 (needed for std::filesystem)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)

# ---------- TensorRT backend (requires CUDA + TensorRT) ----------
find_package(CUDA REQUIRED)
add_library(tensorrt_inferencer SHARED src/tensorrt_inferencer.cpp)
target_include_directories(tensorrt_inferencer PRIVATE
  include
  ${CUDA_INCLUDE_DIRS}
  /usr/include/aarch64-linux-gnu   # TensorRT headers on Jetson
)
target_link_libraries(tensorrt_inferencer
  ${CUDA_LIBRARIES}
  nvinfer
  nvonnxparser
  stdc++fs    # std::filesystem
)
install(TARGETS tensorrt_inferencer DESTINATION lib)

# ---------- DUMMY: Dummy backend (no dependencies, for testing) ----------
# DUMMY: Remove this target before publication
add_library(dummy_inferencer SHARED src/dummy_inferencer.cpp)
target_include_directories(dummy_inferencer PRIVATE include)
install(TARGETS dummy_inferencer DESTINATION lib)
# DUMMY: End dummy backend

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  #set(ament_cmake_copyright_FOUND TRUE)
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# Export and install headers so other packages can #include "inferencer/..."
install(DIRECTORY include/
  DESTINATION include
)

ament_export_include_directories(include)

ament_package()
