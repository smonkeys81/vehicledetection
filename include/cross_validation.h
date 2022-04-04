// Copyright © 2019 Robotics, Inc. All Rights Reserved.

// This file provides methods for cross-validation for vehicle detection.


#ifndef ROBOTICS_VEHICLEDETECTION_CROSSVALIDATION_H_
#define ROBOTICS_VEHICLEDETECTION_CROSSVALIDATION_H_


#include "param/crossval_param.h"
#include "dataset_KITTI.h"
#include "performance_eval.h"
#include <sys/stat.h> 


namespace robotics {
namespace vehicle_detector {
  

// Forward declaration for unit test.
class CrossValidationTest;
  
/// \class CrossValidation
/// This is a class to provide cross validation methods for vehicle detection.
class CrossValidation : public CrossValParam {
friend class PerformanceEvalTest;
public:
  /// \brief Constructor.
  CrossValidation();
  
  /// \brief Destructor.
  ~CrossValidation() {}
  
  /// \brief Execute cross validation procedure.
  /// \param[in] val_config_file Path to validation configuration file.
  /// \param[in] model_name Name of the network backbone model (ZF/VGG16/RESNET101).
  /// \return True if all precedures ran successfully.
  bool CrossValidationDatasetKITTI(const std::string& val_config_file,
                                   const std::string& model_name);

  /// \brief Split dataset into training set and validation set.
  /// \param[in] dataset Dataset.
  /// \param[in] num_k The number of folds.
  /// \param[in] trial Number of attempts. 
  /// \param[out] idx_training Training set list.
  /// \param[out] idx_validation Validation set list.
  /// \return True if all precedures ran successfully.
  bool SplitData(const std::vector<DataImageLabel> dataset,
                 const unsigned int num_k, const unsigned int trial,
                 std::vector<int> &idx_training, std::vector<int> &idx_validation);
  
  /// \brief Find the training configuration file (from input solver file), and update the training label's path or image directory path with the input line (specifying new file for cross validation)
  /// \param[in] solver_file Path to solver file.
  /// \param[in] pattern Pattern to find: "source" or "root_folder".
  /// \param[in] line_input Text to write in the file as a path to labels.
  /// \param[out] line_original Original text of label file field.
  /// \return True if the file is updated succesfully.
  bool UpdateTrainInfo(const std::string& solver_file,
                       const std::string& pattern,
                       const std::string& line_input,
                       std::string& line_original);
public:
  /// \brief KITTI dataset.
  DataSetKITTI kitti_;
};
  
  
} // namespace vehicle_detector
} // namespace robotics


#endif // ROBOTICS_VEHICLEDETECTION_CROSSVALIDATION_H_
