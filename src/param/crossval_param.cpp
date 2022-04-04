// Copyright © 2019 Robotics, Inc. All Rights Reserved.

// This file provides parameters and its loading method for cross validation.


#include "param/crossval_param.h"

namespace robotics {
namespace vehicle_detector {

  
// Load evaluation parameters from file. 
bool CrossValParam::LoadParamCrossVal(const std::string& file) {
  // Open Json file - error/exception message is printed in the function.
  Json::Value root;  
  if(!OpenFileJSON(file, root)) {
    return false;
  }
  
  // K value.
  num_k_ = root["k"].asInt();
  
  // Configuration.
  config_file_ = root["config_file"].asString();

  // Path to dataset.
  dir_label_ = root["path"]["dir_label"].asString();
  dir_image_ = root["path"]["dir_image"].asString();
  dir_image_right_ = root["path"]["dir_image_right"].asString();
 
  // For conversion.
  file_conv_bash_ = root["convert"]["bash_file"].asString();
  num_iter_from_ = root["convert"]["iteration_from"].asInt();
  num_iter_to_ = root["convert"]["iteration_to"].asInt();
  num_iter_step_ = root["convert"]["iteration_step"].asInt();
  
  // For evaluation.
  trained_file_ = root["evaluation"]["trained_file"].asString();
  plot_file_ = root["evaluation"]["plot_file"].asString();
  
  return true;
}
  
  
} // namespace vehicle_detector
} // namespace robotics
