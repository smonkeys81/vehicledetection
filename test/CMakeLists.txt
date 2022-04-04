# Copyright © 2019 Robotics, Inc. All Rights Reserved.

# Copy test model related files.
set(TEST_MODEL_DIR ${PROJECT_SOURCE_DIR}/test/sample_model)
file(COPY ${TEST_MODEL_DIR} DESTINATION ${TEST_OUTPUT_DIR})
set(TEST_IMG_DIR ${PROJECT_SOURCE_DIR}/test/sample_img)
file(COPY ${TEST_IMG_DIR} DESTINATION ${TEST_OUTPUT_DIR})

# Copy shared files from model directory.
set(TEST_LABEL_FILE ${MODEL_SRC_DIR}/synset_words.txt)
set(TEST_PRETRAIN_FILE ${MODEL_SRC_DIR}/ZF/ZF.v2.caffemodel)
set(TEST_CONFIG_FILE ${MODEL_SRC_DIR}/default_config.json)
set(TEST_INFERENCE_FILE ${MODEL_SRC_DIR}/ZF/ZF_faster_rcnn_test.pt)
configure_file(${TEST_LABEL_FILE} ${TEST_OUTPUT_DIR}/sample_model/synset_words.txt COPYONLY)
configure_file(${TEST_PRETRAIN_FILE} ${TEST_OUTPUT_DIR}/sample_model/ZF.v2.caffemodel COPYONLY)
configure_file(${TEST_INFERENCE_FILE} ${TEST_OUTPUT_DIR}/sample_model/ZF_faster_rcnn_test.pt COPYONLY)

#Create output directory
file(MAKE_DIRECTORY ${TEST_OUTPUT_DIR}/out)

file(GLOB_RECURSE TEST_SRC
  "*_test.cpp"
  "../src/*.cpp"
  )

cuda_add_executable(
  unit_tests
  ${TEST_SRC}
  )

target_link_libraries(
  unit_tests
  gtest_main
  ${JSON_LIB}
  ${Boost_LIBRARIES}
  ${GLOG_LIB}
  ${OpenCV_LIBS}
  ${CAFFE_LIB}
  )

add_test(
  NAME	unit
  COMMAND	${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/unit_tests
  )
