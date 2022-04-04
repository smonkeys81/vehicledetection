// Copyright © 2019 Robotics, Inc. All Rights Reserved.

// This file provides parameters and functions to load the parameters for data augmentation.


#ifndef ROBOTICS_VEHICLEDETECTION_AUGMENTPARAM_H_
#define ROBOTICS_VEHICLEDETECTION_AUGMENTPARAM_H_

#include "util.h"
#include <iostream>
#include <string>

namespace robotics {
namespace vehicle_detector {
  

// Define type of augmentation.
enum ENUM_AUGMENT {
  kHorflip,     // Horizontal flip.
  kTranslation, // Translation.
  kRotation,    // Rotation.
  kScale,       // Scale.
  kBrightness,  // Brightness.
  kBlur,        // Gaussian Blur.
  kNoise,       // Gaussian Noise.
  kCutout,      // Cutout.
  kAugment_num, // Number of augmentations
};
  
/// \class AugmentParam
/// This class consists of settings regarding augmentation.
class AugmentParam {
public:
  /// \brief Constructor.
  AugmentParam() {
    do_ = false;
    prob_ = 0.;
    max_ = 0.;
    min_ = 0.;
    x_max_ = 0.;
    y_max_ = 0.;
    custom_ = 0.;
  }
  
  /// \brief Flag whether augment or not.
  bool do_;
  /// \brief Probability of the augmentation.
  float prob_;
  /// \brief Maximum value for the augmentation: intensity, degree, pixels, and etc.
  float max_;
  /// \brief Minimum value for the augmentation: intensity, degree, pixels, and etc.
  float min_;
  /// \brief Maximum x value the augmentation: pixels, and etc.
  float x_max_;
  /// \brief Minimum value for the augmentation: pixels, and etc.
  float y_max_;
  /// \brief Customized value for each augmenting function.
  float custom_;
};
  
// Forward declaration for unit test.
class DataAugmentParamTest;

/// \class DataAugmentParam
/// This is a class to provide augmenting functions to input dataset.
class DataAugmentParam {
friend class DataAugmentParamTest;
public:
  /// \brief Constructor.
  DataAugmentParam();
  
  /// \brief Destructor.
  ~DataAugmentParam();
  
  /// \brief Load parameters to augment images.
  /// \param[in] file Path to configuration file.
  /// \return True if configuration file was loaded successfully, false otherwise.
  bool LoadParamAugment(const std::string& file);
  
public:
  /// \brief Augmentation parameters.
  AugmentParam aug_param_[kAugment_num];  
};
  
  
} // namespace vehicle_detector
} // namespace robotics


#endif // ROBOTICS_VEHICLEDETECTION_AUGMENTPARAM_H_