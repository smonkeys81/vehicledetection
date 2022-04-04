// Copyright © 2019 Robotics, Inc. All Rights Reserved.

// This file provides functions to augment KITTI dataset.


#ifndef ROBOTICS_VEHICLEDETECTION_DATAAUGMENTATION_H_
#define ROBOTICS_VEHICLEDETECTION_DATAAUGMENTATION_H_

#include "dataset_KITTI.h"
#include "param/augment_param.h"
#include "vehicle_detector.h"
#include <iostream>
#include <string>


namespace robotics {
namespace vehicle_detector {
  
  
// Forward declaration for unit test.
class DataAugmentationTest;

/// \class DataAugmentation
/// This is a class to provide augmenting functions to input dataset.
class DataAugmentation : public VehicleDetector, public DataAugmentParam {
friend class DataAugmentationTest;
public:
  /// \brief Constructor.
  /// \param[in] config_file Path to configuration file.
  DataAugmentation(std::string config_file);
  
  /// \brief Destructor.
  ~DataAugmentation() {}
  
  /// \brief Delete all previously augmented images and labels from the specified directories.
  /// \return True if data augmentation was done successfully, false otherwise.
  bool AugmentData();
  
  /// \brief Delete all previously augmented images and labels from the specified directories.
  /// \param[in] path_label Path to label directory.
  /// \param[in] path_image Path to image directory.  
  /// \return True if all files were deleted successfully, false otherwise.
  bool RemoveAugmentedFiles(const std::string& path_label, const std::string& path_image);
  
  /// \brief Perform data augmentation on the specified directory.
  /// \param[in] path_label Path to label directory.
  /// \param[in] path_image Path to image directory.  
  /// \return True if all files were generated and saved successfully, false otherwise.
  bool GenerateAugmentedFiles(const std::string& path_label,
                              const std::string& path_image);
  
  /// \brief Image manipulation - translate.
  /// \param[in] img Input image to manipulate.
  /// \param[in] label Original labels, including GT and don't-care bboxes.
  /// \param[out] gt_bboxes Updated bboxes of vehicle and don't-care region.
  /// \param[out] file_name File name added with parameters.
  /// \return True if there's no exception.
  bool ImageTranslate(cv::Mat& img, const DataImageLabel& label,
                      std::vector<Vehicle>& gt_bboxes,
                      std::string& file_name);
  
  /// \brief Rotate bbox.
  /// \param[in] center Center x and y of image.
  /// \param[in] deg Amount of rotation.
  /// \param[out] box Updated bbox.
  void BoxRotate(const cv::Point_<float> center,
                 const float deg, cv::Rect_<float>& box);
  
  /// \brief Image manipulation - rotate.
  /// \param[in] img Input image to manipulate.
  /// \param[in] label Original labels, including GT and don't-care bboxes.
  /// \param[out] gt_bboxes Updated bboxes of vehicle and don't-care region.
  /// \param[out] file_name File name added with parameters.
  /// \return True if there's no exception.
  bool ImageRotate(cv::Mat& img, const DataImageLabel& label,
                  std::vector<Vehicle>& gt_bboxes,
                  std::string& file_name);
  
  /// \brief Image manipulation - scale.
  /// \param[in] img Input image to manipulate.
  /// \param[in] label Original labels, including GT and don't-care bboxes.
  /// \param[out] gt_bboxes Updated bboxes of vehicle and don't-care region.
  /// \param[out] file_name File name added with parameters.
  /// \return True if there's no exception.
  bool ImageScale(cv::Mat& img, DataImageLabel& label,
                  std::vector<Vehicle>& gt_bboxes,
                  std::string& file_name);
  
  /// \brief Image manipulation - brightness.
  /// \param[in] img Input image to manipulate.
  /// \param[in] label Original labels, including GT and don't-care bboxes.
  /// \param[out] gt_bboxes Updated bboxes of vehicle and don't-care region.
  /// \param[out] file_name File name added with parameters.
  /// \return True if there's no exception.
  bool ImageBrightness(cv::Mat& img, DataImageLabel& label,
                       std::vector<Vehicle>& gt_bboxes,
                       std::string& file_name);
  
  /// \brief Image manipulation - blurring.
  /// \param[in] img Input image to manipulate.
  /// \param[in] label Original labels, including GT and don't-care bboxes.
  /// \param[out] gt_bboxes Updated bboxes of vehicle and don't-care region.
  /// \param[out] file_name File name added with parameters.
  /// \return True if there's no exception.
  bool ImageBlur(cv::Mat& img, DataImageLabel& label,
                       std::vector<Vehicle>& gt_bboxes,
                       std::string& file_name);
  
  /// \brief Image manipulation - Gaussian noise.
  /// \param[in] img Input image to manipulate.
  /// \param[in] label Original labels, including GT and don't-care bboxes.
  /// \param[out] gt_bboxes Updated bboxes of vehicle and don't-care region.
  /// \param[out] file_name File name added with parameters.
  /// \return True if there's no exception.
  bool ImageNoise(cv::Mat& img, DataImageLabel& label,
                  std::vector<Vehicle>& gt_bboxes,
                  std::string& file_name);
  
  /// \brief Image manipulation - cutout.
  /// \param[in] img Input image to manipulate.
  /// \param[in] label Original labels, including GT and don't-care bboxes.
  /// \param[out] gt_bboxes Updated bboxes of vehicle and don't-care region.
  /// \param[out] file_name File name added with parameters.
  /// \return True if there's no exception.
  bool ImageCutout(cv::Mat& img, DataImageLabel& label,
                   std::vector<Vehicle>& gt_bboxes,
                   std::string& file_name);
public:
  /// \brief KITTI Dataset.
  DataSetKITTI kitti_;
};
  
  
} // namespace data_augmentor
} // namespace robotics


#endif // ROBOTICS_VEHICLEDETECTION_DATA_AUGMENTATION_H_
