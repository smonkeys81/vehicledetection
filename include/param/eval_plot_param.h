// Copyright © 2019 Robotics, Inc. All Rights Reserved.

// This file provides plotting parameters and its loading method for performance evaluation.


#ifndef ROBOTICS_VEHICLEDETECTION_EVALPLOTPARAM_H_
#define ROBOTICS_VEHICLEDETECTION_EVALPLOTPARAM_H_


#include "util.h"


namespace robotics {
namespace vehicle_detector {


// Forward declaration for unit test.
class EvalPlotParamTest;


/// \class EvalParamPlot
/// This class consists of arguments and loading method for evaluation.
class EvalPlotParam {
friend class EvalPlotParamTest;
public:
  /// \brief Constructor.
  EvalPlotParam() {}
  
  /// \brief Destructor.
  ~EvalPlotParam() {}

  /// \brief Load evaluation parameters from file. 
  /// \param[in] file path to the configuration file.
  /// \return True if file and the values were loaded successfully.
  bool LoadParamEvalPlot(const std::string& file);
  
public:
  // For line plot.
  /// \brief Line plot image width.
  unsigned int plot_line_size_w_;
  /// \brief Line plot image height.
  unsigned int plot_line_size_h_;
  /// \brief Inner plot size in the line plot image.
  unsigned int plot_line_size_in_;
  /// \brief Shift amount of the x-axis text from the original position.
  int plot_line_pos_x_[2];
  /// \brief Shift amount of the y-axis text from the original position.
  int plot_line_pos_y_[2];
  /// \brief Shift amount of the "prec" text from the original position.
  int plot_line_pos_prec_[2];  
  /// \brief Shift amount of the "recall" text from the original position.
  int plot_line_pos_recall_[2];
  /// \brief Shift amount of the mAP text from the original position.
  int plot_line_pos_ap_[2];
  /// \brief Shift amount of the title text from the original position.
  int plot_line_pos_title_[2];
  
  // For scatter plot.
  /// \brief Scatter plot image width.
  unsigned int plot_sct_size_w_;
  /// \brief Scatter plot image height.
  unsigned int plot_sct_size_h_;
  /// \brief Bottom margin in the scatter plot image.
  unsigned int plot_sct_margin_btm_;
  /// \brief Scale meter to pixel.
  unsigned int plot_sct_scale_;
  /// \brief Distance to divide near and far regions.
  unsigned int plot_sct_dist_near_;
  /// \brief Maximum distance to draw.
  unsigned int plot_sct_dist_max_;
  /// \brief Distance interval to draw.
  unsigned int plot_sct_dist_interval_;
  /// \brief Distance interval to draw.
  unsigned int plot_sct_length_axis_;
  /// \brief Scatter plot - color - background.
  cv::Scalar plot_sct_color_bg_;
  /// \brief Scatter plot - color - thick line.
  cv::Scalar plot_sct_color_line_thick_;
  /// \brief Scatter plot - color - thin line.
  cv::Scalar plot_sct_color_line_thin_;
  /// \brief Scatter plot - color - True positive.
  cv::Scalar plot_sct_color_TP_;
  /// \brief Scatter plot - color - False negative.
  cv::Scalar plot_sct_color_FN_;
  /// \brief Scatter plot - color - False positive.
  cv::Scalar plot_sct_color_FP_;
  /// \brief Scatter plot - color - Dont care.
  cv::Scalar plot_sct_color_DC_;
  /// \brief Shift amount of the x-axis text from the original position.
  int plot_sct_pos_x_[2];
  /// \brief Shift amount of the z-axis text from the original position.
  int plot_sct_pos_z_[2];
  /// \brief Shift amount of the unit text from the original position.
  int plot_sct_pos_unit_[2];
  /// \brief Shift amount of the info1 text from the original position.
  int plot_sct_pos_info1_[2];
  /// \brief Shift amount of the info2 text from the original position.
  int plot_sct_pos_info2_[2];

  /// \brief Amount of the row space.
  int plot_txt_space_row_;
  
  /// \brief Font - size.
  float font_size_;
  /// \brief Font - thickness.
  float font_thickness_;
  /// \brief Font - space to adjust position of the text in the image.
  unsigned int font_space_;
  
  /// \brief Color - background.
  cv::Scalar bg_color_;
  /// \brief Color - inner background.
  cv::Scalar inner_bg_color_;
  /// \brief Color - grid line.
  cv::Scalar grid_line_color_;
  /// \brief Color - marker.
  cv::Scalar mark_color_;
  
  /// \brief Image save range - min score.
  float img_save_score_min_;
  /// \brief Image save range - max score.
  float img_save_score_max_;
  /// \brief Image save range - interval.
  unsigned int img_save_interval_;
};
  
  
} // namespace vehicle_detector
} // namespace robotics


#endif // ROBOTICS_VEHICLEDETECTION_EVALPLOTPARAM_H_