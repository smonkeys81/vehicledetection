// Copyright © 2019 Robotics, Inc. All Rights Reserved.

// This file provides a tool to call cross-validation method for vehicle detection.

#include "cross_validation.h"
#include <chrono>


// Define arguments.
enum ENUM_ARGUMENT {
  kConfig = 1,
  kNetModel,
  kNumArg,
};

/// \brief Entry function for evaluation. 
/// \param[in] argc Number of arguments.
/// \param[in] argv Character array of arguments.
/// \return True if all procedures were successful.
int main(int argc, char** argv) {
  // Exception handlign - check the number of arguments.
  if (argc != kNumArg) {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0]
      << " [configuration].json"
      << " Network_Model_Name(ZF/VGG16/RESNET101)"
      << std::endl;
    return 0;
  }
  
  // Measure time - start.
  auto t_start = std::chrono::high_resolution_clock::now();
  std::cout << "Start cross validation for KITTI" << std::endl;
  std::cout << "================================" << std::endl;
  // Store arguments.
  // Each path is verified when it is used in a function.
  const std::string config_file = argv[kConfig];
  std::string model_name = argv[kNetModel];
  
  // Validation instance.
  robotics::vehicle_detector::CrossValidation val;
  
  // Call cross validation method.
  if(!val.CrossValidationDatasetKITTI(config_file, model_name)) {
    std::cerr << "Error occurred during validation" << std::endl;
    return 0; 
  }
  
  // Measure time - end.
  auto t_end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::seconds>( t_end - t_start ).count();
  std::cout << "================================================" << std::endl;
  std::cout << "End cross validation for KITTI (Total duration: " 
    << duration << "s)" <<std::endl;
  
  return 1;
}