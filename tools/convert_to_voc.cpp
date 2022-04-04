// Copyright © 2020 Robotics, Inc. All Rights Reserved.

// This file provides methods to convert labels of KITTI/BDD dataset into VOC labels.

#include "vehicle_detector.h"
#include "dataset_KITTI.h"
#include "dataset_BDD100K.h"
#include <opencv2/highgui/highgui.hpp>

// Define arguments.
enum ENUM_ARGUMENT {
  kDataset = 1,
  kInputLabel,
  kOutputLabel,
  kNumArg,
};

/// \brief Entry function to convert the KITTI into VOC. 
/// \param[in] argc Number of arguments.
/// \param[in] argv Character array of arguments.
/// \return True if all procedures were successful.
int main(int argc, char** argv) {
  // Exception handlign - check the number of arguments.
  if (argc != kNumArg) {
    std::cerr << "Usage: " << std::endl
      << argv[0]
      << " [Dataset (kitti or bdd)]"
      << " [path to original label(s) of dataset]"
      << " [path and file name to write].trainval" << std::endl;
    return 0;
  }
  
  // Store arguments.
  std::string dataset_input = argv[kDataset];
  const std::string path_label = argv[kInputLabel];
  const std::string file_label = argv[kOutputLabel];
  
  // Make upper case.
  std::transform(dataset_input.begin(), dataset_input.end(), dataset_input.begin(), ::toupper);
  
  std::cout << "Start converting labels from " << dataset_input << " to VOC" << std::endl;
  
  const int num_kitti = robotics::vehicle_detector::kKITTI;
  const int num_bdd = robotics::vehicle_detector::kBDD100K;
  const std::string str_kitti = robotics::vehicle_detector::dataset_str[num_kitti];
  const std::string str_bdd = robotics::vehicle_detector::dataset_str[num_bdd];
  
  if(!dataset_input.compare(str_kitti)) {
    robotics::vehicle_detector::DataSetKITTI kitti;
    if(!kitti.ConvertVOC(path_label, file_label)) {
      std::cerr << "Unable to convert data." << std::endl;
      return 0;
    }
  } else if(!dataset_input.compare(str_bdd)) {
    robotics::vehicle_detector::DataSetBDD100K bdd;
    if(!bdd.ConvertVOC(path_label, file_label)) {
      std::cerr << "Unable to convert data." << std::endl;
      return 0;
    }
  } else {
    std::cerr << "Invalid dataset name: " << dataset_input << std::endl;
    return 0;
  }
  
  std::cout << "End converting labels from " << dataset_input << " to VOC" << std::endl;
  
  return 1;
}
