# Copyright © 2019 Robotics, Inc. All Rights Reserved.

# Find required packages
find_package(Boost 1.54 REQUIRED COMPONENTS system thread filesystem)
find_package(CUDA REQUIRED)
find_package(OpenCV 3.0.0 REQUIRED)
# Master version of caffe doesn't work for faster RCNN
find_package(Caffe REQUIRED)
set(CAFFE_INCLUDE_DIRS ${Caffe_DIR}/install/include)
find_library(CAFFE_LIB NAMES caffe HINTS ${Caffe_DIR}/lib)

# multi GPU with Nvidia NCCL
set(NCCL_INC_PATHS
    /usr/include
    /usr/local/include
    $ENV{NCCL_DIR}/include
    )
set(NCCL_LIB_PATHS
    /lib
    /lib64
    /usr/lib
    /usr/lib64
    /usr/local/lib
    /usr/local/lib64
    $ENV{NCCL_DIR}/lib
    )
find_path(NCCL_INCLUDE_DIR NAMES nccl.h PATHS ${NCCL_INC_PATHS})
find_library(NCCL_LIBRARIES NAMES nccl PATHS ${NCCL_LIB_PATHS})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL DEFAULT_MSG NCCL_INCLUDE_DIR NCCL_LIBRARIES)

if (NCCL_FOUND)
  message(STATUS "Found NCCL    (include: ${NCCL_INCLUDE_DIR}, library: ${NCCL_LIBRARIES})")
  #add_definitions(-DUSE_NCCL)
endif ()

# For json
set(JSON_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/thirdparty/json/include)
set(JSON_LIB ${PROJECT_SOURCE_DIR}/thirdparty/json/lib/libjsoncpp.a)

# Msg - name and path
message("$Boost_INCLUDE_DIRS= " ${Boost_INCLUDE_DIRS})
message("$Boost_LIBRARIES= " ${Boost_LIBRARIES})
message("$CUDA_INCLUDE_DIRS= " ${CUDA_INCLUDE_DIRS})
message("$CUDA_LIBRARIES= " ${CUDA_LIBRARIES})
message("$Caffe_DIR= " ${Caffe_DIR})
message("$CAFFE_INCLUDE_DIRS= " ${CAFFE_INCLUDE_DIRS})
message("$CAFFE_LIB= " ${CAFFE_LIB})
message("$OpenCV_INCLUDE_DIRS= " ${OpenCV_INCLUDE_DIRS})
message("$OpenCV_LIBS= " ${OpenCV_LIBS})
message("$JSON_INCLUDE_DIRS= " ${JSON_INCLUDE_DIRS})
message("$JSON_LIB= " ${JSON_LIB})

# For header files
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CAFFE_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${JSON_INCLUDE_DIRS})

# Set Glog file
set(GLOG_LIB libglog.so.0)
# TODO(H.Lee) - check if there's another way to find glog
