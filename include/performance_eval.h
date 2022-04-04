// Copyright © 2019 Robotics, Inc. All Rights Reserved.

// This file provides performance evaluating functionalities to the model used in vehicle detection pipeline.


#ifndef ROBOTICS_VEHICLEDETECTION_PERFORMANCEEVAL_H_
#define ROBOTICS_VEHICLEDETECTION_PERFORMANCEEVAL_H_


#include "vehicle_detector.h"
#include "dataset_KITTI.h"
#include "dataset_BDD100K.h"
#include "param/eval_plot_param.h"


namespace robotics {
namespace vehicle_detector {


// Define distance zone in birdeye view.
enum ENUM_DISTANCE {
  kNear,
  kFar,
  kNum_Dist,
};
static const char *distance_str[] = {"Near", "Far"};
  
// Forward declaration for unit test.
class PerformanceEvalTest;

/// \class EvalDataLoc
/// This class consists of locations and its information (TP, TN or FP).
class EvalDataLoc {
public:
  /// \brief Calculate distance from ego-vehicle to the object.
  /// \return Distance.
  float GetDist() const { return sqrt(loc_3D_[k3D_X] * loc_3D_[k3D_X]
                                      + loc_3D_[k3D_Z] * loc_3D_[k3D_Z]); }
public:
  /// \brief 3D coordinates.
  float loc_3D_[k3D_num];
  /// \brief Bbox.
  cv::Rect_<float> bbox_;
  /// \brief Difficulty.
  int difficulty_;
  /// \brief Result.
  int result_;
  /// \brief Test file's name.
  std::string file_name_;
};
  
/// \class EvalDataPrecRec
/// This class consists of precision, recall and corresponding results.
class EvalDataPrecRec {
public:
  /// \brief Recall.
  float recall_;
  /// \brief Precision.
  float prec_;  
  /// \brief Score threshold.
  float thres_;
  /// \brief Recall and precision value. 
  std::vector<EvalDataLoc> data_loc_;
};
  
/// \class EvalDataMAP
/// This class contains all result data for the specified IoU.
class EvalDataMAP {
public:
  /// \brief IoU.
  float iou_;
  /// \brief mAP.
  float mAP_;
  /// \brief Recall and precision value. 
  std::vector<EvalDataPrecRec> prec_recall_;
};
  
/// \class PerformanceEval
/// This is a class to provide performance evaluation of vehicle detection.
class PerformanceEval : public VehicleDetector, public EvalPlotParam {
friend class PerformanceEvalTest;
public:
  /// \brief Constructor.
  /// \param[in] config_file Path to configuration file.
  PerformanceEval(std::string config_file);
  
  /// \brief Destructor.
  ~PerformanceEval() {}
  
  /// \brief Test all images in the dataset.
  /// \param[in] set Index of dataset, e.g., 0: KITTI, 1: BDD100K. 
  /// \param[in] debug_msg Print debug message. 
  /// \return True if all images were tested, false otherwise.
  bool EvaluateDataset(const unsigned int set, const bool debug_msg);
  
  /// \brief Count true positive from ground-truth vehicles and predicted vehicles.
  /// \param[in] gt Ground-truth vehicles.
  /// \param[in] dontcare Ignored bboxes.
  /// \param[in] pred Predicted vehicles.
  /// \param[in] data_loc 3D location data.
  /// \param[in] overlap_thres Threshold value of minimum overlap.
  /// \param[in] file_name File name of the input image.
  /// \param[out] num_true_pos Number of true positives.
  /// \param[out] num_dont_care Number of dont_cares.
  /// \return True if counting was done successfully.
  bool CountTP(std::vector<Vehicle>& gt, 
              std::vector<Vehicle>& dontcare, 
              std::vector<Vehicle>& pred,
              std::vector<EvalDataLoc>& data_loc, 
              unsigned int& num_true_pos,
              unsigned int& num_dont_care,
              const float overlap_thres,
              const std::string& file_name);
  
  /// \brief Save cropped bbox image of TP, FN and FP.
  /// \param[in] set Index of dataset, e.g., 0: KITTI, 1: BDD100K. 
  /// \param[in] idx_iou Index to IoU in all data.
  /// \param[in] idx_thres Index to precision and recall in all data.
  /// \return True if all crop image save successfully, false otherwise.
  bool SaveBbox(const unsigned int set, 
                const int idx_iou, const int idx_thres);
  
  /// \brief Draw background of line plot.
  /// \param[in] img Image for plotting.
  void VisualizePlotLineBG(cv::Mat img);
  
  /// \brief Show evaluation result. 
  void VisualizePlotLineResult();
  
  /// \brief Draw background of radius plot.
  /// \param[in] img Image for plotting.
  /// \param[in] max_dist Maximum distance to plot.
  void VisualizePlotRadBG(cv::Mat img, const unsigned int max_dist);
  
  /// \brief Visualize detection results over camera radius.
  /// \param[out] idx_iou Index to IoU.
  /// \param[out] idx_thres Index to precision and recall.
  void VisualizePlotRadResult(int& idx_iou, int& idx_thres);
  
  /// \brief Load models and all parameters needed to perform evaluation.
  /// \param[in] model_name Name of the backbone model.
  /// \param[in] trained_file Trained file including parameter values.
  /// \return True if all loading procedure is completed successfully, false otherwise.
  bool LoadAll(const std::string& model_name,
               const std::string& trained_file);
public:
  /// \brief KITTI Dataset.
  DataSetKITTI kitti_;
  
  /// \brief BDD100K Dataset.
  DataSetBDD100K bdd_;
  
  /// \brief All result data.
  std::vector<EvalDataMAP> data_all_;
};
  
  
} // namespace vehicle_detector
} // namespace robotics


#endif // ROBOTICS_VEHICLEDETECTION_PERFORMANCEEVAL_H_
