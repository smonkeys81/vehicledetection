// Copyright © 2019 Robotics, Inc. All Rights Reserved.

// This file provides methods to augment dataset.

#include "data_augmentation.h"
#include <chrono>


// Define arguments.
enum ENUM_ARGUMENT {
  kConfig = 1,
  kNumArg,
};

/// \brief Entry function for data augmentation. 
/// \param[in] argc Number of arguments.
/// \param[in] argv Character array of arguments.
/// \return True if all procedures were successful.
int main(int argc, char** argv) {
  // Exception handlign - check the number of arguments.
  if (argc != kNumArg) {
    std::cerr << "Usage: " << std::endl
      << argv[0] 
      << " [config].json" << std::endl;
    return 0;
  }

  std::cout << "Start dataset augmentation for KITTI" << std::endl;
    
  // Measure time - start.
  auto t_start = std::chrono::high_resolution_clock::now();
  
  // Store arguments.
  // Each path is verified when it is used in a function.
  const std::string config_file = argv[kConfig];
  
  // Evaluation instance.
  robotics::vehicle_detector::DataAugmentation augmentor(config_file);
    
  if(!augmentor.AugmentData()) {
    std::cerr << "Augmentation process failed" << std::endl;
    return 0;
  }

  // Measure time - end.
  auto t_end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::seconds>( t_end - t_start ).count();
  std::cout << "End dataset augmentation for KITTI (Total duration: " 
    << duration << "s)" <<std::endl;
  
  return 1;	
}
