// Copyright © 2019 Robotics, Inc. All Rights Reserved.

// This file provides methods to evaluate the performance of the trained model for vehicle detection.

#include "performance_eval.h"
#include <chrono>
#include <utility>
#include <vector>
#include <json.h>


// Define arguments.
enum ENUM_ARGUMENT {
  kDataset = 1,
  kConfig,
  kNetModel,
  kWeight,
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
      << " [Dataset (kitti or bdd)]"
      << " [config].json"
      << " Network_Model_Name(ZF/VGG16/RESNET101)"
      << " [training(weight)].caffemodel"
      << std::endl;
    return 0;
  }

  // Measure time - start.
  auto t_start = std::chrono::high_resolution_clock::now();
  
  // Store arguments.
  // Each path is verified when it is used in a function.
  const std::string dataset_input =
    robotics::vehicle_detector::MakeUpperCase(argv[kDataset]);
  const std::string config_file = argv[kConfig];
  const std::string model_name = 
    robotics::vehicle_detector::MakeUpperCase(argv[kNetModel]);
  const std::string trained_file = argv[kWeight];
  
  // Evaluation instance.
  robotics::vehicle_detector::PerformanceEval eval(config_file);
    
  // Load model and configurations of evaluation and plotting.
  if(!eval.LoadAll(model_name, trained_file)) {
    // Exception messages are generated inside the function.
    return 0;
  }
  
  const int num_kitti = robotics::vehicle_detector::kKITTI;
  const int num_bdd = robotics::vehicle_detector::kBDD100K;
  const std::string str_kitti = robotics::vehicle_detector::dataset_str[num_kitti];
  const std::string str_bdd = robotics::vehicle_detector::dataset_str[num_bdd];
  
  // Load all labels and images.
  std::cout << "Loading test dataset... " << std::endl;
  
  int num_data;
  if(!dataset_input.compare(str_kitti)) {
    num_data = num_kitti;    
  } else if(!dataset_input.compare(str_bdd)) {
    num_data = num_bdd;
  } else {
    std::cerr << "failed" << std::endl;
    return 0;
  }
  
  // If birdeye and disparity results are not required,
  // we can skip loading right images.
  const std::string dir_label = eval.param_.dir_dataset_test_label_[num_data];
  const std::string dir_image = eval.param_.dir_dataset_test_image_[num_data];
  const std::string dir_image_r = eval.param_.dir_dataset_test_image_right_[num_data];
  
  if(num_data == num_kitti) {
    if(!eval.param_.save_image_disparity_ && !eval.param_.save_image_birdeye_) {
      if(!eval.kitti_.LoadDataSet(dir_label, dir_image)) {
      std::cerr << "failed" << std::endl;
      return 0;
      }
    } else {
      if(!eval.kitti_.LoadDataSet(dir_label, dir_image, dir_image_r)) {
        std::cerr << "failed" << std::endl;
        return 0;
      }
    }
  } else {
    if(!eval.bdd_.LoadDataSet(dir_label, dir_image)) {
        std::cerr << "failed" << std::endl;
        return 0;
      }
  }
  
  
  // Call evaluating function.
  std::cout << "Start performance evaluation for KITTI" << std::endl;
  
  std::cout << "======================================================" << std::endl;
  std::time_t result = std::time(nullptr);
  std::cout << std::asctime(std::localtime(&result));
   
  // Call evaluating function.
  if(!eval.EvaluateDataset(num_data, false)) {
    std::cerr << "Error occurred during evaluation" << std::endl;
    return 0;
  }    

  std::cout << "======================================================" << std::endl;
  
  // Measure time - end.
  auto t_end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::seconds>( t_end - t_start ).count();
  std::cout << "End performance evaluation for KITTI (Total duration: " 
    << duration << "s)" <<std::endl;
  
  // Visualize and save results.
  if(eval.data_all_.size() > 0) {
    eval.VisualizePlotLineResult();
    // Index to data.
    int idx_iou = -1;
    int idx_thres = -1;
    eval.VisualizePlotRadResult(idx_iou, idx_thres);
    // Bboxes are saved after saving birdeye view to find the optimal threshold.
    if(!eval.SaveBbox(num_data, idx_iou, idx_thres)) {
      std::cerr << "Error has occurred." << std::endl;
      return false;
    }
  }  
  
  return 1;	
}
