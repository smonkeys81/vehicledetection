// Copyright © 2019 Robotics, Inc. All Rights Reserved.

// This file provides plotting parameters and its loading method for performance evaluation.


#include "param/eval_plot_param.h"

namespace robotics {
namespace vehicle_detector {

  
// Load evaluation parameters from file. 
bool EvalPlotParam::LoadParamEvalPlot(const std::string& file) {
  // Open Json file - error/exception message is printed in the function.
  Json::Value root;  
  if(!OpenFileJSON(file, root)) {
    return false;
  }
  
  // For color info.
  int r, g, b, grey;
  
  // For Line plot.
  // Image size.
  plot_line_size_w_ = root["line_plot"]["img_size_w"].asInt();
  plot_line_size_h_ = root["line_plot"]["img_size_h"].asInt();
  // Inner plot size in the image.
  plot_line_size_in_ = root["line_plot"]["plot_size"].asInt();
  // Shift text.
  plot_line_pos_x_[0] = root["line_plot"]["pos"]["axis_x"][0].asInt();
  plot_line_pos_x_[1] = root["line_plot"]["pos"]["axis_x"][1].asInt();
  plot_line_pos_y_[0] = root["line_plot"]["pos"]["axis_y"][0].asInt();
  plot_line_pos_y_[1] = root["line_plot"]["pos"]["axis_y"][1].asInt();
  plot_line_pos_prec_[0] = root["line_plot"]["pos"]["prec"][0].asInt();
  plot_line_pos_prec_[1] = root["line_plot"]["pos"]["prec"][1].asInt();
  plot_line_pos_recall_[0] = root["line_plot"]["pos"]["recall"][0].asInt();
  plot_line_pos_recall_[1] = root["line_plot"]["pos"]["recall"][1].asInt();
  plot_line_pos_ap_[0] = root["line_plot"]["pos"]["ap"][0].asInt();
  plot_line_pos_ap_[1] = root["line_plot"]["pos"]["ap"][1].asInt();
  plot_line_pos_title_[0] = root["line_plot"]["pos"]["title"][0].asInt();
  plot_line_pos_title_[1] = root["line_plot"]["pos"]["title"][1].asInt();
  
  // For Scattered plot.
  // Image size.
  plot_sct_size_w_ = root["scatter_plot"]["img_size_w"].asInt();
  plot_sct_size_h_ = root["scatter_plot"]["img_size_h"].asInt();
  plot_sct_margin_btm_ = root["scatter_plot"]["margin_bottom"].asInt();
  plot_sct_scale_ = root["scatter_plot"]["scale"].asInt();
  plot_sct_dist_near_ = root["scatter_plot"]["dist_near"].asInt();
  plot_sct_dist_max_ = root["scatter_plot"]["dist_max"].asInt();
  plot_sct_dist_interval_ = root["scatter_plot"]["dist_interval"].asInt();
  plot_sct_length_axis_ = root["scatter_plot"]["axis_length"].asInt();
  r = root["scatter_plot"]["color"]["bg"][0].asInt();
  g = root["scatter_plot"]["color"]["bg"][1].asInt();
  b = root["scatter_plot"]["color"]["bg"][2].asInt();
  plot_sct_color_bg_ = CV_RGB(r, g, b);
  r = root["scatter_plot"]["color"]["line_thick"][0].asInt();
  g = root["scatter_plot"]["color"]["line_thick"][1].asInt();
  b = root["scatter_plot"]["color"]["line_thick"][2].asInt();
  plot_sct_color_line_thick_ = CV_RGB(r, g, b);
  r = root["scatter_plot"]["color"]["line_thin"][0].asInt();
  g = root["scatter_plot"]["color"]["line_thin"][1].asInt();
  b = root["scatter_plot"]["color"]["line_thin"][2].asInt();
  plot_sct_color_line_thin_ = CV_RGB(r, g, b);
  r = root["scatter_plot"]["color"]["TruePos"][0].asInt();
  g = root["scatter_plot"]["color"]["TruePos"][1].asInt();
  b = root["scatter_plot"]["color"]["TruePos"][2].asInt();
  plot_sct_color_TP_ = CV_RGB(r, g, b);
  r = root["scatter_plot"]["color"]["FalseNeg"][0].asInt();
  g = root["scatter_plot"]["color"]["FalseNeg"][1].asInt();
  b = root["scatter_plot"]["color"]["FalseNeg"][2].asInt();
  plot_sct_color_FN_ = CV_RGB(r, g, b);
  r = root["scatter_plot"]["color"]["FalsePos"][0].asInt();
  g = root["scatter_plot"]["color"]["FalsePos"][1].asInt();
  b = root["scatter_plot"]["color"]["FalsePos"][2].asInt();
  plot_sct_color_FP_ = CV_RGB(r, g, b);
  r = root["scatter_plot"]["color"]["Dontcare"].asInt();
  plot_sct_color_DC_ = CV_RGB(r, r, r);
  plot_sct_pos_x_[0] = root["scatter_plot"]["pos"]["axis_x"][0].asInt();
  plot_sct_pos_x_[1] = root["scatter_plot"]["pos"]["axis_x"][1].asInt();
  plot_sct_pos_z_[0] = root["scatter_plot"]["pos"]["axis_z"][0].asInt();
  plot_sct_pos_z_[1] = root["scatter_plot"]["pos"]["axis_z"][1].asInt();
  plot_sct_pos_unit_[0] = root["scatter_plot"]["pos"]["unit"][0].asInt();
  plot_sct_pos_unit_[1] = root["scatter_plot"]["pos"]["unit"][1].asInt();
  plot_sct_pos_info1_[0] = root["scatter_plot"]["pos"]["info1"][0].asInt();
  plot_sct_pos_info1_[1] = root["scatter_plot"]["pos"]["info1"][1].asInt();
  plot_sct_pos_info2_[0] = root["scatter_plot"]["pos"]["info2"][0].asInt();
  plot_sct_pos_info2_[1] = root["scatter_plot"]["pos"]["info2"][1].asInt();
  
  plot_txt_space_row_ = root["txt_space_row"].asInt();
  
  // Font.
  font_size_ = root["font"]["size"].asFloat();
  font_thickness_ = root["font"]["thickness"].asFloat();
  font_space_ = root["font"]["space"].asInt();
  // Color.
  grey = root["color"]["bg"].asInt();
  bg_color_ = CV_RGB(grey, grey, grey);
  grey = root["color"]["inner_bg"].asInt();
  inner_bg_color_ = CV_RGB(grey, grey, grey);
  grey = root["color"]["grid_line"].asInt();
  grid_line_color_ = CV_RGB(grey, grey, grey);
  r = root["color"]["mark"][0].asInt();
  g = root["color"]["mark"][1].asInt();
  b = root["color"]["mark"][2].asInt();
  mark_color_ = CV_RGB(r, g, b);
  
  

  // Image save.
  img_save_score_min_ = root["image_save_score_min"].asFloat();
  img_save_score_max_ = root["image_save_score_max"].asFloat();
  img_save_interval_ = root["image_save_interval"].asInt();

  return true;
}
  
  
} // namespace vehicle_detector
} // namespace robotics