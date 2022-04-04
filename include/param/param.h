// Copyright © 2020 Robotics, Inc. All Rights Reserved.

// This file provides general parameters for vehicle detection pipeline.


#ifndef ROBOTICS_VEHICLEDETECTION_PARAM_H_
#define ROBOTICS_VEHICLEDETECTION_PARAM_H_

#include "util.h"
#include <iostream>


namespace robotics {
namespace vehicle_detector {
  

// Define dataset string.
enum ENUM_DATASET {
  kKITTI = 0,
  kBDD100K,
  kNum_Dataset,
};
static const char *dataset_str[] = {"KITTI", "BDD"};

// Define backbone string.
enum ENUM_BACKBONE {
  kNet_ZF = 0,
  kNet_VGG16,
  kNet_Res101,
  kNet_FPN,
  kNum_Backbone,
};
static const char *backbone_str[] = {"ZF", "VGG16", "RES101", "FPN"};

// Define detection type.
enum ENUM_RESULT_CLASS {
  kTruePos,  // true positive
  kFalseNeg,  // false negative
  kFalsePos,  // false positive
  kDontCare,  // Dont-care: none of above.
  kRESULT_num,
};

// Define max number of colored line.
const int max_color_line=10;
  
// Forward declaration for unit test.
class ParamTest;

/// \class DataAugmentParam
/// This is a class to provide augmenting functions to input dataset.
class Param {
friend class ParamTest;
public:
  /// \brief Constructor.
  Param() {}
  
  /// \brief Destructor.
  ~Param() {}
  
  /// \brief Load parameters to augment images.
  /// \param[in] file Path to configuration file.
  /// \return True if configuration file was loaded successfully, false otherwise.
  bool LoadParam(const std::string& file);
  
  /// \brief Retrieve the index of backbone from the input string.
  /// \param[in] model_name Name of the network backbone model (ZF/VGG16/RESNET101).
  /// \param[out] index_model Index of backbone model.
  /// \return True if the index is found, false otherwise.
  bool GetBackboneNum(const std::string& model_name, unsigned int& index_model);
  
public: 
  /// \brief GPU number to use.
  unsigned int device_gpu_num_;
  
  /// \brief root directory to load dataset.
  std::string dir_dataset_root_[kNum_Dataset];
  /// \brief directory for training images.
  std::string dir_dataset_train_image_[kNum_Dataset];
  /// \brief directory for training labels.
  std::string dir_dataset_train_label_[kNum_Dataset];
  /// \brief directory for test images.
  std::string dir_dataset_test_image_[kNum_Dataset];
  /// \brief directory for test images, if stereo images are provided.
  std::string dir_dataset_test_image_right_[kNum_Dataset];
  /// \brief directory for test images, if stereo images are provided.
  std::string dir_dataset_test_label_[kNum_Dataset];
  /// \brief root directory to save result files.
  std::string dir_result_;
  
  /// \brief root directory for files.
  std::string file_root_;
  /// \brief path for network configuration file.
  std::string file_net_config_;
  /// \brief path for label definition file.
  std::string file_net_label_;
  /// \brief path for data augmentation configuration.
  std::string file_data_augmentation_;
  /// \brief path for plot configuration.
  std::string file_plotting_;
  /// \brief path for solver file.
  std::string file_solver_[kNum_Backbone];
  /// \brief path for pretrained weights file.
  std::string file_pretrain_[kNum_Backbone];
  /// \brief path for test network.
  std::string file_net_test_[kNum_Backbone];
  
  /// \brief Score iteration.
  int eval_iter_score_;
  /// \brief Iteration from the lowest score (iter_score_).
  int eval_iter_score_low_;
  /// \brief Camera FOV.
  float eval_cam_fov_;
  /// \brief Camera f-number.
  float eval_cam_fnum_;
  /// \brief Camera baseline.
  float eval_cam_baseline_;  
  /// \brief IoU - start.
  float eval_iou_start_;
  /// \brief IoU - end.
  float eval_iou_end_;
  /// \brief IoU - interval.
  float eval_iou_step_;
  /// \brief IoU number of line color.
  int eval_iou_num_color_;
  /// \brief IoU color definitions.
  cv::Scalar eval_line_color_[max_color_line];
  
  /// \brief convolution image - save directory.
  std::string save_image_conv_dir_;
  /// \brief convolution image - Min number of columns for merged image.
  unsigned int save_image_conv_num_col_merged_from_;
  /// \brief convolution image - Max number of columns for merged image.
  unsigned int save_image_conv_num_col_merged_to_;
  /// \brief convolution image - width boundary to determine if image is big.
  unsigned int save_image_conv_large_width_;
  
  /// \brief birdeye view image - flag.
  bool save_image_birdeye_;
  /// \brief birdeye view image - save directory.
  std::string save_image_birdeye_dir_;
  /// \brief birdeye view image - iou.
  float save_image_birdeye_iou_;
  /// \brief birdeye view image - target precision value.
  float save_image_birdeye_prec_target_;
  /// \brief birdeye view image - weight of precision.
  float save_image_birdeye_prec_weight_;
  /// \brief birdeye view image - target recall value.
  float save_image_birdeye_recall_target_;
  /// \brief birdeye view image - weight of recall.
  float save_image_birdeye_recall_weight_;
  /// \brief birdeye view image - number of samples to save in sub-images.
  unsigned int save_image_birdeye_num_subsamples_;
  
  /// \brief overlay image - flag.
  bool save_image_overlay_;
  /// \brief overlay image - save directory.
  std::string save_image_overlay_dir_;
  /// \brief overlay image - minimum score to save.
  float save_image_overlay_score_min_;
  /// \brief overlay image - maximum score to save.
  float save_image_overlay_score_max_;
  /// \brief overlay image - inteval between score for saving image.
  unsigned int save_image_overlay_interval_;

  /// \brief disparity image - flag.
  bool save_image_disparity_;
  /// \brief disparity image - save directory.
  std::string save_image_disparity_dir_;
  /// \brief disparity image - minimum score to draw box.
  float save_image_disparity_score_min_;
  
  /// \brief bbox image - whether save images or not.
  bool save_image_bbox_;
  /// \brief bbox image - whether save TP images or not.
  bool save_image_bbox_tp_;
  /// \brief bbox image - save directory for TP, FN, FP box images.
  std::string save_image_bbox_dir_[kRESULT_num];
  /// \brief bbox image - IoU value to save.
  float save_image_bbox_iou_;
  /// \brief bbox image - Score value (threshold) to save.
  float save_image_bbox_score_;

  /// \brief Font scale.
  double font_scale_;
  /// \brief Font thickness.
  double font_thickness_;
  /// \brief Font row space.
  double font_row_space_;

  /// \brief Stereo matching - mode.
  unsigned int stereo_mode_;
  /// \brief Stereo matching - window size.
  unsigned int stereo_win_size_;
  /// \brief Stereo matching - range of disparities.
  unsigned int stereo_num_disparities_;
  /// \brief Stereo matching - block size.
  unsigned int stereo_block_size_;
  /// \brief Stereo matching - smoothing factor - p1.
  unsigned int stereo_smooth_p1_;
  /// \brief Stereo matching - smoothing factor - p2.
  unsigned int stereo_smooth_p2_;  
};
  
  
} // namespace vehicle_detector
} // namespace robotics


#endif // ROBOTICS_VEHICLEDETECTION_PARAM_H_
