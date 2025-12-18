cmake_minimum_required(VERSION 3.8)
project(inferencer)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)

include_directories(include)

# Always build mock plugin
add_library(mock_inferencer SHARED
  src/mock_inferencer.cpp
)
target_include_directories(mock_inferencer PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(mock_inferencer ament_index_cpp)
target_link_libraries(mock_inferencer dl)
set_target_properties(mock_inferencer PROPERTIES OUTPUT_NAME "mock_inferencer")

# Optional TensorRT plugin
find_library(NVINFER_LIB nvinfer)
find_library(NVONNXPARSER_LIB nvonnxparser)
find_library(CUDART_LIB cudart)

if(NVINFER_LIB AND NVONNXPARSER_LIB AND CUDART_LIB)
  add_library(tensorrt_inferencer SHARED
    src/tensorrt_inferencer.cpp
  )
  target_include_directories(tensorrt_inferencer PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )
  target_link_libraries(tensorrt_inferencer
    ${NVINFER_LIB}
    ${NVONNXPARSER_LIB}
    ${CUDART_LIB}
    dl
  )
  set_target_properties(tensorrt_inferencer PROPERTIES OUTPUT_NAME "tensorrt_inferencer")
else()
  message(WARNING "TensorRT backend disabled (missing libs: nvinfer/nvonnxparser/cudart). This is OK on non-Jetson machines.")
endif()

install(TARGETS
  mock_inferencer
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

if(TARGET tensorrt_inferencer)
  install(TARGETS tensorrt_inferencer
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib/${PROJECT_NAME}
  )
endif()

install(DIRECTORY include/
  DESTINATION include
)

ament_export_include_directories(include)
ament_export_dependencies(ament_index_cpp)
ament_package()
