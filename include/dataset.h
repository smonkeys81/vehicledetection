// Copyright © 2019 Robotics, Inc. All Rights Reserved.

// This file defines components of dataset and loading method.


#ifndef ROBOTICS_VEHICLEDETECTION_DATASET_H_
#define ROBOTICS_VEHICLEDETECTION_DATASET_H_


#include "vehicle_detector.h"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

#define VOC_LABEL_CAR 7


namespace robotics {
namespace vehicle_detector {
  
  
// Defining stereo camara system.
enum ENUM_CAM {
  kCam_Left = 0,
  kCam_Right,
  kCam_Max,
};
  
// Defining stereo matching method.
enum ENUM_STEREO_MATCHING {
  kStereo_BM = 0,
  kStereo_SGBM,
};
  
// Define how to describe coordinate.
enum ENUM_COORD {
  kTopLeftBottomRight = 0,
  kTopLeftWidthHeight,
};
  
/// \class DataImageLabel
/// This class consists of an image and its bbox information.
class DataImageLabel {
public:
  /// \brief Images.
  cv::Mat img_[kCam_Max];
  /// \brief File name.
  std::string file_name_;
  /// \brief Container holding ground-truth bboxes, don't care bboxes, and predictions.
  std::vector<Vehicle> gt_vehicles_, gt_dontcare_, detection_;
};
  
// Forward declaration for unit test.
class DataSetTest;
  
/// \class DataSet
/// This class consists of dataset of an image and its bbox information.
class DataSet {
friend class DataSetTest;
public:
  /// \brief Constructor.
  DataSet() { exist_img_right_ = false; }
  
  /// \brief Destructor.
  ~DataSet() {}
  
  /// \brief Virtual function to load dataset.
  /// \return True.
  virtual bool LoadDataSet() {};

  /// \brief Estimate 3d location of the prediction by using stereo images.
  /// \param[in] param Parameters for stereo matching.
  /// \param[in] idx Index of the image from dataset.
  /// \param[in] debug Print debugging messages and save image.
  /// \param[in] save_dir Directory path to save images.
  /// \return True if estimation processed was done successfully, false otherwise.
  bool Estimate3DLoc(const Param param,
                     const unsigned int idx,
                     const bool debug = false,
                     const std::string& save_dir = "");
  
  /// \brief Calculate average of intensity values within the input box area. 0 values are not counted as data.
  /// \param[in] img Disparity image.
  /// \param[in] box Bbox in the image.
  /// \return Avaerage intensity of the bbox.
  float CalculateAvgIntensity(cv::Mat img, cv::Rect box);
  
  /// \brief Calculate angle of the vehicle from the ego-vehicle.
  /// \param[in] h_fov_deg Horizontal FOV of the camera in degree.
  /// \param[in] img_width Width of the image in pixel.
  /// \param[in] box Bbox in the image.
  /// \return Angle of the vehicle in degree.
  float CalculateAngle(const float h_fov_deg, const unsigned int img_width,
                       cv::Rect box);
  
  /// \brief Write GT-vehicles into the VOC output file.
  /// \param[in] file_label File path to save output.
  /// \param[in] vehicles GT-vehicles.
  /// \param[in] seq Sequential number of image from 0.
  /// \param[in] img_file_name File name of the image.  
  /// \param[in] mode Description how to represent coordinates.
  void SaveVOC(std::ofstream& file_label, const std::vector<Vehicle>& vehicles,
               unsigned int& seq, std::string img_file_name, const int mode);
public:
  /// \brief Dataset.
  std::vector<DataImageLabel> dataset_;
  
  /// \brief Flag for existence of right images.
  bool exist_img_right_;
};
  
  
} // namespace vehicle_detector
} // namespace robotics


#endif // ROBOTICS_VEHICLEDETECTION_DATASET_H_
