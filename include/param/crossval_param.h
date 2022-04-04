// Copyright © 2019 Robotics, Inc. All Rights Reserved.

// This file provides parameters and its loading method for cross validation.


#ifndef ROBOTICS_VEHICLEDETECTION_CROSSVALPARAM_H_
#define ROBOTICS_VEHICLEDETECTION_CROSSVALPARAM_H_


#include "util.h"


namespace robotics {
namespace vehicle_detector {


// Forward declaration for unit test.
class CrossValParamTest;


/// \class CrossValParam
/// This class consists of arguments and loading method for evaluation.
class CrossValParam {
friend class CrossValParamTest;
public:
  /// \brief Constructor.
  CrossValParam() {}
  
  /// \brief Destructor.
  ~CrossValParam() {}

  /// \brief Load evaluation parameters from file. 
  /// \param[in] file path to the configuration file.
  /// \return True if file and the values were loaded successfully.
  bool LoadParamCrossVal(const std::string& file);
  
public:
  /// \brief k value for k-fold cross validation.
  unsigned int num_k_;
  /// \brief Configuration file.
  std::string config_file_;
  /// \brief Label directory.
  std::string dir_label_;
  /// \brief Image directory.
  std::string dir_image_;
  /// \brief Right images directory.
  std::string dir_image_right_;

  // For conversion.
  /// \brief Path to bash file for conversion from trained weights file to test file.
  std::string file_conv_bash_;
  /// \brief Number of iterations to begin.
  unsigned int num_iter_from_;
  /// \brief Number of iterations to end.
  unsigned int num_iter_to_;
  /// \brief Interval between iterations.
  unsigned int num_iter_step_;
  
  // For evaluation.
  /// \brief Trained weights file.
  std::string trained_file_;
  /// \brief Plotting configuration file.
  std::string plot_file_;
};
  
  
} // namespace vehicle_detector
} // namespace robotics


#endif // ROBOTICS_VEHICLEDETECTION_CROSSVALPARAM_H_
