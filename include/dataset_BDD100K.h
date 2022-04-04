// Copyright © 2020 Robotics, Inc. All Rights Reserved.

// This file provides definitions regarding BDD100K dataset.
// https://bair.berkeley.edu/blog/2018/05/30/bdd/

#ifndef ROBOTICS_VEHICLEDETECTION_DATASETBDD100K_H_
#define ROBOTICS_VEHICLEDETECTION_DATASETBDD100K_H_


#include "util.h"
#include "dataset.h"


namespace robotics {
namespace vehicle_detector {

  
/// \class DataSet
/// This class consists of dataset of an image and its bbox information.
class DataSetBDD100K : public DataSet {
public:
  /// \brief Load all images from the dataset.
  /// \param[in] path_label Directory path containing labels.
  /// \param[in] path_img Directory path containing left images from stereo camera.
  /// \return True if all images were loaded successfully, false otherwise.
  bool LoadDataSet(const std::string& path_label,
                   const std::string& path_img);
  
  /// \brief Convert the BDD100K data into VOC format.
  /// \param[in] in_file Input file path containing labels.
  /// \param[in] out_file File path to save output.
  /// \return True if all data were converted successfully, false otherwise.
  bool ConvertVOC(const std::string& in_file,
                   const std::string& out_file);
};
  
  
} // namespace vehicle_detector
} // namespace robotics


#endif // ROBOTICS_VEHICLEDETECTION_DATASETBDD100K_H_