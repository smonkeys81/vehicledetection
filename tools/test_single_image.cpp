// Copyright © 2019 Robotics, Inc. All Rights Reserved.

// This file provides methods to test single image.

#include "vehicle_detector.h"
#include <opencv2/highgui/highgui.hpp>

// Define arguments.
enum ENUM_ARGUMENT {
  kConfig = 1,
  kNetModel,
  kWeight,
  kPathImg,
  kNumArg,
};

/// \brief Entry function for single image test. 
/// \param[in] argc Number of arguments.
/// \param[in] argv Character array of arguments.
/// \return True if all procedures were successful.
int main(int argc, char** argv) {
  // Exception handlign - check the number of arguments.
  if (argc != kNumArg) {
    std::cerr << "Usage: " << std::endl
      << argv[0]
      << " [config].json"
      << " Network_Model_Name(ZF/VGG16/RESNET101)"
      << " [training(weight)].caffemodel"
      << " [path to image]"
      << std::endl;
    return 0;
  }
  
  // Store arguments.
  // Each path is verified when it is used in a function.
  const std::string config_file = argv[kConfig];
  const std::string model_name =
    robotics::vehicle_detector::MakeUpperCase(argv[kNetModel]);
  const std::string trained_file = argv[kWeight];
  const std::string path_image = argv[kPathImg];
    
  // Evaluation instance.
  robotics::vehicle_detector::VehicleDetector detector(config_file);
  
  // Load model.
  std::cout << "Load Model... ";
  if(!detector.LoadModelByName(model_name, trained_file)) {
    std::cerr << "failed" << std::endl;
    return 0;
  }
  std::cout << "complete" << std::endl;

  // Load image.
  if(!robotics::vehicle_detector::FileExist(path_image)) {
    std::cerr << __FUNCTION__ << " Unable to open image file." << std::endl;
    return 0;
  }
  cv::Mat img = cv::imread(path_image, -1);
  
  // Call test function.
  std::vector<robotics::vehicle_detector::Vehicle> detections;
  if(!detector.Detect(img, true, detections)) {
    std::cerr << "Error occurred during detection." << std::endl;
    return 0;
  }
  
  // Visualize detections.
  detector.VisualizeDetections(img, detections,
                               robotics::vehicle_detector::kBbox_pred,
                               robotics::vehicle_detector::kBbox_Score);
  
  // Create default result directory.
  std::string path = detector.param_.dir_result_ 
    + robotics::vehicle_detector::PathSeparator();
  mkdir(path.c_str(), 0777);  
  
  // Save detection image.        
  cv::imwrite(path + "detection.png", img);
  
  // Create sub-directory.
  path += detector.param_.save_image_conv_dir_ 
    + robotics::vehicle_detector::PathSeparator();
  mkdir(path.c_str(), 0777);
  
  // Save all convolution images.
  detector.SaveAllConvolutionImages(path);
  
  return 1;	
}
