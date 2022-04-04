# Copyright © 2019 Robotics, Inc. All Rights Reserved.

# Collect source files
file(GLOB_RECURSE srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE TOOL_SRC "../src/*.cpp")

# Build each source file independently
foreach(source ${srcs})
  get_filename_component(name ${source} NAME_WE)

  # target
  cuda_add_executable(
    ${name} 
    ${source} 
    ${TOOL_SRC}
    )

  target_link_libraries(
    ${name}
    ${JSON_LIB}
    ${Boost_LIBRARIES}
    ${GLOG_LIB}
    ${OpenCV_LIBS}
    ${CAFFE_LIB}
    )

endforeach(source)
