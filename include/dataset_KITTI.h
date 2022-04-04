// Copyright © 2019 Robotics, Inc. All Rights Reserved.

// This file provides definitions regarding KITTI dataset.

#ifndef ROBOTICS_VEHICLEDETECTION_DATASETKITTI_H_
#define ROBOTICS_VEHICLEDETECTION_DATASETKITTI_H_


#include "util.h"
#include "dataset.h"

#define KITTI_VEHICLE_CAR "Car"
#define KITTI_VEHICLE_VAN "Van"
#define KITTI_VEHICLE_TRUCK "Truck"
#define KITTI_DONTCARE "DontCare"
#define KITTI_MISC "Misc"
  

namespace robotics {
namespace vehicle_detector {


// File name pattern to distinguish augmented files from the original dataset.
const std::string aug = "_aug_";
  
// Define KITTI dataset format.
enum ENUM_KITTI {
  kKITTI_type = 0,  // Car, Van, Truck, Pedestrian, Person_sitting, Cyclist, Tram, Misc, DontCare.
  kKITTI_truncated, // 0: non-truncated, 1: truncated.
  kKITTI_occluded,  // 0: fully visibly, 1: partly occluded, 2: largely occluded, 3: unknown.
  kKITTI_alpha, // Observation angle [-pi, pi].
  kKITTI_bbox_left, // Bounding box in pixel.
  kKITTI_bbox_top,  // Bounding box in pixel.
  kKITTI_bbox_right,  // Bounding box in pixel.
  kKITTI_bbox_bottom, // Bounding box in pixel.
  kKITTI_3d_height, // Object dimension in meters.
  kKITTI_3d_width,  // Object dimension in meters.
  kKITTI_3d_length, // Object dimension in meters.
  kKITTI_3d_loc_x,  // Object location in camera coordinate in meters.
  kKITTI_3d_loc_y,  // Object location in camera coordinate in meters.
  kKITTI_3d_loc_z,  // Object location in camera coordinate in meters.
  kKITTI_score,  // Confidence in detection.
  kKITTI_num_data,
};
  
/// \class DataSet
/// This class consists of dataset of an image and its bbox information.
class DataSetKITTI : public DataSet {
public:
  /// \brief Load all images from the dataset.
  /// \param[in] path_label Directory path containing labels.
  /// \param[in] path_img_left Directory path containing left images from stereo camera (This dir is mendatory).
  /// \param[in] path_img_right Directory path containing right images from stereo camera (This dir is optional for evaluation).
  /// \return True if all images were loaded successfully, false otherwise.
  bool LoadDataSet(const std::string& path_label,
                   const std::string& path_img_left,
                   const std::string& path_img_right = "");
  
  /// \brief Convert the KITTI data into VOC format.
  /// \param[in] path_label Directory path containing labels.
  /// \param[in] file_label File path to save output.
  /// \return True if all data were converted successfully, false otherwise.
  bool ConvertVOC(const std::string& path_label,
                   const std::string& file_label);
};
  
  
} // namespace vehicle_detector
} // namespace robotics


#endif // ROBOTICS_VEHICLEDETECTION_DATASETKITTI_H_