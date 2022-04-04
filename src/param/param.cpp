// Copyright © 2019 Robotics, Inc. All Rights Reserved.

// This file provides general parameters for vehicle detection pipeline.


#include "param/param.h"

namespace robotics {
namespace vehicle_detector {

   
// Load evaluation parameters from file. 
bool Param::LoadParam(const std::string& file) {
  // Open Json file - error/exception message is printed in the function.
  Json::Value root;  
  if(!OpenFileJSON(file, root)) {
    return false;
  }
  
  // Set device info.
  device_gpu_num_ = root["device"]["gpu"].asInt();
  
  // Directory paths - multiple dataset can be loaded.
  const Json::Value dataset = root["directory"]["dataset"];
  for(int i = 0; i < dataset.size(); ++i) {
    // Get the name of dataset.
    std::string dataset_name = root["directory"]["dataset"][i]["name"].asString();
    
    MakeUpperCase(dataset_name);
    
    for(int j = 0; j < kNum_Dataset; ++j) {
      if(!dataset_name.compare(dataset_str[j])) {
        dir_dataset_root_[j] = root["directory"]["dataset"][j]["root"].asString();
        dir_dataset_train_image_[j] = dir_dataset_root_[j] 
          + root["directory"]["dataset"][i]["train_image"].asString();
        dir_dataset_train_label_[j] = dir_dataset_root_[j] 
          + root["directory"]["dataset"][i]["train_label"].asString();
        dir_dataset_test_image_[j] = dir_dataset_root_[j] 
          + root["directory"]["dataset"][i]["test_image"].asString();
        dir_dataset_test_image_right_[j] = dir_dataset_root_[j] 
          + root["directory"]["dataset"][i]["test_image_right"].asString();
        dir_dataset_test_label_[j] = dir_dataset_root_[j] 
          + root["directory"]["dataset"][i]["test_label"].asString();
      }
    }
  }
  dir_result_ = root["directory"]["result"].asString();
  
  // Configuration/definition files.
  file_root_ = root["file"]["root"].asString();
  file_net_config_ = file_root_ + root["file"]["net_config"].asString();
  file_net_label_ = file_root_ + root["file"]["net_label"].asString();
  file_data_augmentation_ = file_root_ + root["file"]["data_augmentation"].asString();
  file_plotting_ = file_root_ + root["file"]["plotting"].asString();
  // Read CNN backbone files.
  const Json::Value backbone = root["file"]["backbone"];
  for(int i = 0; i < backbone.size(); ++i) {
    // Get backbone name.
    std::string backbone_name = root["file"]["backbone"][i]["name"].asString();
    
    // Make upper case.
    MakeUpperCase(backbone_name);
    
    for(int j = 0; j < kNum_Backbone; ++j) {
      if(!backbone_name.compare(backbone_str[j])) {
        // Solver.
        file_solver_[j] = file_root_ + backbone_name + PathSeparator() 
          + root["file"]["backbone"][i]["solver"].asString();
        // Pretrain - optional.
        // If string is empty, assign empty string.
        const std::string pretrain = root["file"]["backbone"][i]["pretrain"].asString();
        if(pretrain == "") {
          file_pretrain_[j] = pretrain;
        } else {
          file_pretrain_[j] = file_root_ + backbone_name + PathSeparator() 
          + root["file"]["backbone"][i]["pretrain"].asString();
        }        
        // Test.
        file_net_test_[j] = file_root_ + backbone_name + PathSeparator()
          + root["file"]["backbone"][i]["net_test"].asString();
      }
    }
  }
  
  // Eval - score.
  eval_iter_score_ = root["eval"]["iter_score"].asInt();
  eval_iter_score_low_ = root["eval"]["iter_score_low"].asInt();
  // Eval - Camera.
  eval_cam_fov_ = root["eval"]["camera"]["fov"].asFloat();
  eval_cam_fnum_ = root["eval"]["camera"]["f"].asFloat();
  eval_cam_baseline_ = root["eval"]["camera"]["baseline"].asFloat();
  // Eval - IoU.
  eval_iou_start_ = root["eval"]["iou"]["start"].asFloat();
  eval_iou_end_ = root["eval"]["iou"]["end"].asFloat();
  eval_iou_step_ = root["eval"]["iou"]["step"].asFloat();
  eval_iou_num_color_ = root["eval"]["iou"]["num_color"].asInt();
  for(int i = 0; i < eval_iou_num_color_; ++i) {
    int r, g, b;
    std::string line = "line" + std::to_string(i);
    r = root["eval"]["iou"]["color"][line][0].asInt();
    g = root["eval"]["iou"]["color"][line][1].asInt();
    b = root["eval"]["iou"]["color"][line][2].asInt();
    eval_line_color_[i] = CV_RGB(r, g, b);
  }
  
  // Image save - convolution images.
  save_image_conv_dir_ = root["save_image"]["conv"]["dir"].asString();
  save_image_conv_num_col_merged_from_ = root["save_image"]["conv"]["num_col_merged_from"].asInt();
  save_image_conv_num_col_merged_to_ = root["save_image"]["conv"]["num_col_merged_to"].asInt();
  save_image_conv_large_width_ = root["save_image"]["conv"]["large_width"].asInt(); 
  // Image save - birdeye images.
  save_image_birdeye_ = root["save_image"]["birdeye"]["save"].asBool();
  save_image_birdeye_dir_ = root["save_image"]["birdeye"]["dir"].asString();
  save_image_birdeye_iou_ = root["save_image"]["birdeye"]["iou"].asFloat();
  save_image_birdeye_prec_target_ = root["save_image"]["birdeye"]["prec_target"].asFloat();
  save_image_birdeye_prec_weight_ = root["save_image"]["birdeye"]["prec_weight"].asFloat();
  save_image_birdeye_recall_target_ = root["save_image"]["birdeye"]["recall_target"].asFloat();
  save_image_birdeye_recall_weight_ = root["save_image"]["birdeye"]["recall_weight"].asFloat();
  save_image_birdeye_num_subsamples_ = root["save_image"]["birdeye"]["num_sub_samples"].asInt();
  // Image save - overlay images.
  save_image_overlay_ = root["save_image"]["overlay"]["save"].asBool();
  save_image_overlay_dir_ = root["save_image"]["overlay"]["dir"].asString();
  save_image_overlay_score_min_ = root["save_image"]["overlay"]["score_min"].asFloat();
  save_image_overlay_score_max_ = root["save_image"]["overlay"]["score_max"].asFloat();
  save_image_overlay_interval_ = root["save_image"]["overlay"]["interval"].asInt();
  // Image save - disparity images.
  save_image_disparity_ = root["save_image"]["disparity"]["save"].asBool();
  /// \brief disparity image - save directory.
  save_image_disparity_dir_ = root["save_image"]["disparity"]["dir"].asString();
  save_image_disparity_score_min_ = root["save_image"]["disparity"]["score_min"].asFloat();
  // Image save - bbox images.
  save_image_bbox_ = root["save_image"]["bbox"]["save"].asBool();
  save_image_bbox_tp_ = root["save_image"]["bbox"]["save_tp"].asBool();
  save_image_bbox_dir_[kTruePos] = root["save_image"]["bbox"]["dir_tp"].asString();
  save_image_bbox_dir_[kFalseNeg] = root["save_image"]["bbox"]["dir_fn"].asString();
  save_image_bbox_dir_[kFalsePos] = root["save_image"]["bbox"]["dir_fp"].asString();
  save_image_bbox_iou_ = root["save_image"]["bbox"]["iou"].asFloat();
  save_image_bbox_score_ = root["save_image"]["bbox"]["score"].asFloat();
  
  // Text and font.
  font_scale_ = root["font"]["size"].asDouble();
  font_thickness_ = root["font"]["thickness"].asDouble();
  font_row_space_ = root["font"]["row_space"].asInt();

  // Stereo matching.
  stereo_mode_ = root["stereo_matching"]["mode"].asInt();
  stereo_win_size_ = root["stereo_matching"]["win_size"].asInt();
  stereo_num_disparities_ = root["stereo_matching"]["num_disparities"].asInt();
  stereo_block_size_ = root["stereo_matching"]["block_size"].asInt();
  stereo_smooth_p1_ = root["stereo_matching"]["smooth_p1"].asInt();
  stereo_smooth_p2_ = root["stereo_matching"]["smooth_p2"].asInt();  

  return true;
}

// Retrieve the index of backbone from the input string.
bool Param::GetBackboneNum(const std::string& model_name, unsigned int& index_model) {
  // Find match.
  bool matched = false;
  for(unsigned int i = 0; i < kNum_Backbone; ++i) {
    if(!model_name.compare(backbone_str[i])) {
      index_model = i;
      matched = true;
    }
  }
  
  return matched;
}
  
  
} // namespace vehicle_detector
} // namespace robotics
