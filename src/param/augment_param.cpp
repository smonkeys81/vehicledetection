// Copyright © 2019 Robotics, Inc. All Rights Reserved.

// This file provides parameters and functions to load the parameters for data augmentation.


#include "param/augment_param.h"

namespace robotics {
namespace vehicle_detector {

  
// Constructor.
DataAugmentParam::DataAugmentParam() {
}
  
// Destructor.
DataAugmentParam::~DataAugmentParam() {
}
  
// Load evaluation parameters from file. 
bool DataAugmentParam::LoadParamAugment(const std::string& file) {
  // Open Json file - error/exception message is printed in the function.
  Json::Value root;  
  if(!OpenFileJSON(file, root)) {
    return false;
  }
  
  // Flip.
  aug_param_[kHorflip].do_ = root["Horflip"]["do"].asBool();
  aug_param_[kHorflip].prob_ = root["Horflip"]["prob"].asFloat(); 
  // Translation (padding).
  aug_param_[kTranslation].do_ = root["Translation"]["do"].asBool();
  aug_param_[kTranslation].prob_ = root["Translation"]["prob"].asFloat();
  aug_param_[kTranslation].max_ = root["Translation"]["max"].asFloat();
  aug_param_[kTranslation].min_ = root["Translation"]["min"].asFloat();
  // Rotation.
  aug_param_[kRotation].do_ = root["Rotation"]["do"].asBool();
  aug_param_[kRotation].prob_ = root["Rotation"]["prob"].asFloat();
  aug_param_[kRotation].max_ = root["Rotation"]["max"].asFloat();
  aug_param_[kRotation].min_ = root["Rotation"]["min"].asFloat();
  // Scale.
  aug_param_[kScale].do_ = root["Scale"]["do"].asBool();
  aug_param_[kScale].prob_ = root["Scale"]["prob"].asFloat();
  aug_param_[kScale].max_ = root["Scale"]["max"].asFloat();
  aug_param_[kScale].min_ = root["Scale"]["min"].asFloat();
  // Brightness.
  aug_param_[kBrightness].do_ = root["Brightness"]["do"].asBool();
  aug_param_[kBrightness].prob_ = root["Brightness"]["prob"].asFloat();
  aug_param_[kBrightness].max_ = root["Brightness"]["max"].asFloat();
  aug_param_[kBrightness].min_ = root["Brightness"]["min"].asFloat();
  // Blur.
  aug_param_[kBlur].do_ = root["Blur"]["do"].asBool();
  aug_param_[kBlur].prob_ = root["Blur"]["prob"].asFloat();
  aug_param_[kBlur].max_ = root["Blur"]["max"].asFloat();
  aug_param_[kBlur].min_ = root["Blur"]["min"].asFloat();
  // Noise.
  aug_param_[kNoise].do_ = root["Noise"]["do"].asBool();
  aug_param_[kNoise].prob_ = root["Noise"]["prob"].asFloat();
  aug_param_[kNoise].max_ = root["Noise"]["max"].asFloat();
  aug_param_[kNoise].min_ = root["Noise"]["min"].asFloat();
  // Cutout.
  aug_param_[kCutout].do_ = root["Cutout"]["do"].asBool();
  aug_param_[kCutout].prob_ = root["Cutout"]["prob"].asFloat();
  aug_param_[kCutout].max_ = root["Cutout"]["max"].asFloat();
  aug_param_[kCutout].min_ = root["Cutout"]["min"].asFloat();
  aug_param_[kCutout].custom_ = root["Cutout"]["ratio"].asFloat();
  
  return true;
}
  
  
} // namespace vehicle_detector
} // namespace robotics