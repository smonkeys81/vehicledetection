// Copyright © 2019 Robotics, Inc. All Rights Reserved.

// This file provides methods to train a model.

#include "vehicle_detector.h"

// Define arguments.
enum ENUM_ARGUMENT {
  kConfig = 1,
  kNetModel,
  kSnapshot,
  kNumArg,  
};

/// \brief Entry function for training model. 
/// \param[in] argc Number of arguments.
/// \param[in] argv Character array of arguments.
/// \return True if all procedures were successful.
int main(int argc, char** argv) {
  // Exception handlign - check the number of arguments.
  if (argc != kNumArg && argc != kNumArg - 1) {
    std::cerr << "Usage: " << std::endl
      << argv[0]
      << " [config].json" 
      << " Network_Model_Name(ZF/VGG16/RESNET101)"
      << " [snapshot file].solverstate"
      << std::endl;
    return 0;
  }
  
  // Store arguments.
  const std::string config_file = argv[kConfig];
  const std::string model_name = 
    robotics::vehicle_detector::MakeUpperCase(argv[kNetModel]);
  std::string snapshot = "";
  if(argc == kNumArg) {
    snapshot = argv[kSnapshot];
  }
  std::cout << "Start training" << std::endl;
  
  // Training instance.
  robotics::vehicle_detector::VehicleDetector detector(config_file);
  
  // Train a model.
  if(!detector.TrainModelByName(model_name, snapshot)) {
    std::cerr << "Training failed" << std::endl;
    return 0;
  }
  
  std::cout << "End training" << std::endl;
    
  return 1;
}
