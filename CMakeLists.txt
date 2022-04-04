# Copyright © 2019 Robotics, Inc. All Rights Reserved.

# The official CMake version for Robotics, Inc. is 2.8.
# However, Caffe requires CMake version >= 2.8.7.
# Make is officially supported while CMake is supported by the community. (https://caffe.berkeleyvision.org/installation.html)
cmake_minimum_required(VERSION 2.8.7)

set(CMAKE_CXX_STANDARD 11)

# project name
set(PROJECT_NAME vd_pipeline)
project(${PROJECT_NAME})

# Msg - name and path
message("$PROJECT_NAME= " ${PROJECT_NAME})
message("$PROJECT_SOURCE_DIR= " ${PROJECT_SOURCE_DIR})

# For header files
include_directories(${PROJECT_SOURCE_DIR}/include)

# Additional denpendencies
include(cmake/dependencies.cmake)

# Definitions
set(TEST_OUTPUT_DIR ${CMAKE_BINARY_DIR}/test)
add_definitions(-DPACKAGE_TEST_PATH="${TEST_OUTPUT_DIR}")

# Copy configuration
set(SCRIPT_SRC_DIR ${PROJECT_SOURCE_DIR}/config)
file(COPY ${SCRIPT_SRC_DIR} DESTINATION ${CMAKE_BINARY_DIR})

# Copy scripts
set(SCRIPT_SRC_DIR ${PROJECT_SOURCE_DIR}/scripts)
file(COPY ${SCRIPT_SRC_DIR} DESTINATION ${CMAKE_BINARY_DIR})

# Copy models
set(MODEL_SRC_DIR ${PROJECT_SOURCE_DIR}/model)
file(COPY ${MODEL_SRC_DIR} DESTINATION ${CMAKE_BINARY_DIR})
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/model/out)

# For gtest
include(cmake/googletest.cmake)
fetch_googletest(
  ${PROJECT_SOURCE_DIR}/cmake
  ${PROJECT_BINARY_DIR}/googletest
  )

# Enable testing
enable_testing()

# Define tests
add_subdirectory(test)

# Make tools
add_subdirectory(tools)


