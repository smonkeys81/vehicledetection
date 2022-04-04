// Copyright © 2019 Robotics, Inc. All Rights Reserved.

// This file provides data structure and methods for the vehicle detection pipeline


#ifndef ROBOTICS_VEHICLEDETECTION_VEHICLE_H_
#define ROBOTICS_VEHICLEDETECTION_VEHICLE_H_


#include <opencv2/imgproc/imgproc.hpp>


namespace robotics {
namespace vehicle_detector {


// Define KITTI dataset format.
enum ENUM_3D {
  k3D_X,
  k3D_Y,
  k3D_Z,
  k3D_num,
};
  
// Define difficulty level.
enum ENUM_DIFFICULTY {
  kFull_Visible = 0,
  kOccluded_Partly,
  kOccluded_Largely,
  kDifficulty_Unknown,
};
  
/// \class Vehicle
/// This class contains bounding box(bbox), vehicle type and score of detected vehicle.
class Vehicle {
public:
  /// \brief Constructor.
  Vehicle() { 
    score_ = 0.; 
    loc_3D_[k3D_X] = 0.;
    loc_3D_[k3D_Y] = 0.;
    loc_3D_[k3D_Z] = 0.;
    difficulty_ = kDifficulty_Unknown;
  }

  /// \brief Constructor.
  /// \param[in] score Score [0.0, 1.0] predicted by model.
  /// \param[in] rect Bbox of the vehicle.
  Vehicle(const float score, const cv::Rect_<float> rect) : score_(score), bbox_(rect) {
    loc_3D_[k3D_X] = 0.;
    loc_3D_[k3D_Y] = 0.;
    loc_3D_[k3D_Z] = 0.;
    difficulty_ = kDifficulty_Unknown;
  }

  /// \brief Constructor.
  /// \param[in] score Score [0.0, 1.0] predicted by model.
  /// \param[in] rect Bbox of the vehicle.
  /// \param[in] difficulty Difficulty level of the vehicle instance.
  Vehicle(const float score, const cv::Rect_<float> rect,
          const unsigned int difficulty) : score_(score), bbox_(rect), difficulty_(difficulty) {
    loc_3D_[k3D_X] = 0.;
    loc_3D_[k3D_Y] = 0.;
    loc_3D_[k3D_Z] = 0.;
  }

  /// \brief Destructor.
  ~Vehicle() {}
	
  /// \brief Return score.
  /// \return Score of the prediction.
  float GetScore() const { return score_; }	

  /// \brief Return left coordinate of vehicle bbox.
  /// \return Left coordinate of the vehicle.
  float GetLeft() const { return bbox_.x; }
  
  /// \brief Return right coordinate of vehicle bbox.
  /// \return Right coordinate of the vehicle.
  float GetRight() const { return bbox_.x + bbox_.width; }
  
  /// \brief Return top coordinate of vehicle bbox.
  /// \return Top coordinate of the vehicle.
  float GetTop() const { return bbox_.y; }
  
  /// \brief Return bottom coordinate of vehicle bbox.
  /// \return Bottom coordinate of the vehicle.
  float GetBottom() const { return bbox_.y + bbox_.height; }
  
  /// \brief Return bbox of the vehicle.
  /// \return Bbox of the vehicle.
  cv::Rect_<float> GetBbox() const { return bbox_; }

  /// \brief Return 3D coordinate of the vehicle.
  /// \param[in] idx Index number of coordinate.
  /// \return Coordinate to the corresponding index.
  float Get3DLoc(const ENUM_3D idx) const {return loc_3D_[idx]; }
  
  /// \brief Return name of label.
  /// \return Name of label.
  std::string GetLabel() const { return label_; }
  
  /// \brief Return name of label.
  /// \return Difficulty level of vehicle.
  unsigned int GetDifficulty() const { return difficulty_; }
  
  /// \brief Store score value.
  /// \param[in] score Predicted score to store.
  void SetScore(const float score) { score_ = score; }

  /// \brief Set bbox.
  /// \param[in] rect Bbox of vehicle.
  void SetBbox(const cv::Rect_<float> rect) { bbox_ = rect; }

  /// \brief Set 3D location.
  /// \param[in] x X coordinate of vehicle.
  /// \param[in] y Y coordinate of vehicle.
  /// \param[in] z Z coordinate of vehicle.
  void Set3DLoc(const float x, const float y, const float z)
  { loc_3D_[k3D_X] = x; loc_3D_[k3D_Y] = y; loc_3D_[k3D_Z] = z;}
  
  /// \brief Set label.
  /// \param[in] label Label of vehicle. 
  void SetLabel(const std::string label) { label_ = label; }
  
  /// \brief Set difficulty.
  /// \param[in] difficulty Difficulty level of vehicle. 
  void SetDifficulty(const unsigned int difficulty) { difficulty_ = difficulty; }
private:
  /// \brief Bbox of vehicle.
  cv::Rect_<float> bbox_;

  /// \brief 3D coordinates.
  float loc_3D_[k3D_num];
    
  /// \brief Score predicted for vehicle.
  float score_;
  
  /// \brief Label name of vehicle. e.g., car, truck.
  std::string label_;
  
  /// \brief Difficulty level of the vehicle, depending on the occlusion.
  unsigned int difficulty_;
};


} // namespace vehicle_detector
} // namespace robotics


#endif // ROBOTICS_VEHICLEDETECTION_VEHICLE_H_
