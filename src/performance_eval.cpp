// Copyright © 2019 Robotics, Inc. All Rights Reserved.

// This file provides performance evaluating functionalities to the model used in vehicle detection pipeline.

#include "performance_eval.h"
#include <caffe/FRCNN/util/frcnn_param.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sys/stat.h>


namespace robotics {
namespace vehicle_detector {

  
/// \brief Driver function to sort the vector elements. 
/// \param[in] a one of two elements to be compared.
/// \param[in] b the other one of two to be compared.
/// \return True if a comes first.
bool SortByBoth(const EvalDataPrecRec &a, 
              const EvalDataPrecRec &b) {
  // This procedure enables:
  // 1. Sorting ascending order by first element (recall).
  // 2. Then sorting descending order by second element (precision).
  // This is to find the highest precision value among the same recall.
  if (a.recall_ < b.recall_) {
    return true;
  }
  if (b.recall_ < a.recall_) {
    return false;
  }
  // a=b for primary condition, go to secondary (precision).
  if (a.prec_ > b.prec_) {
    return true;
  }
  if (b.prec_ > a.prec_) {
    return false;
  }
  
  return (a.prec_ < b.prec_); 
} 
  
// Constructor.
PerformanceEval::PerformanceEval(std::string config_file) : VehicleDetector(config_file) {
}
  
// Test all images in the dataset.
bool PerformanceEval::EvaluateDataset(const unsigned int set,
                                      const bool debug_msg = false) {
  // Choose dataset.
  std::vector<DataImageLabel> *ptr_data;
  bool exist_img_right = false;
  if (set == kKITTI) {
    ptr_data = &kitti_.dataset_;
    exist_img_right = kitti_.exist_img_right_;
  } else if (set == kBDD100K) {
    ptr_data = &bdd_.dataset_;
  } else {
    ErrMsg(__func__, "Invalid dataset.");
    return false;    
  }
  const unsigned int size_dataset = ptr_data->size();
  
  // Total duration for detection.
  float duration_total = 0.0;
  // Total Number of vehicles in dataset.
  unsigned int num_vehicle_gt = 0;
  
  // Create default result directory.
  mkdir(param_.dir_result_.c_str(), 0777);
  
  // Directories.
  const std::string dir_disparity = param_.dir_result_ + PathSeparator() 
    + param_.save_image_disparity_dir_;
  const std::string dir_img = param_.dir_result_ + PathSeparator() 
    + param_.save_image_overlay_dir_;
  
  // Create directory if required.
  if(exist_img_right && param_.save_image_disparity_) {
    mkdir(dir_disparity.c_str(), 0777);
  }
  if(param_.save_image_overlay_) {
    mkdir(dir_img.c_str(), 0777);
  }

  // Open dataset and process once.
  for(auto i = 0; i < size_dataset; ++i) {
    // Increase ground-truth vehicle count.
    num_vehicle_gt += ptr_data->at(i).gt_vehicles_.size();
    
    std::vector<Vehicle> detections;
    caffe::Frcnn::FrcnnParam::test_score_thresh = 0.0;
    
    // Measure time - start.
    auto t_start = std::chrono::high_resolution_clock::now();
    // Process.
    bool ret = Detect(ptr_data->at(i).img_[kCam_Left], debug_msg, ptr_data->at(i).detection_);
    // Measure time - end.
    auto t_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t_end - t_start ).count();
          
    if(!ret) {
      ErrMsg(__func__, "Detect process failed.");
      return false;
    }
    
    // Sum duration.
    duration_total += duration / 1000.;  
    
    if(i % 500 == 0 && i > 0) {
      std::cout << i << "/" << size_dataset << " images were processed." << std::endl;
    }
    
    // Estimate locations of the predicted vehicle.
    // To check debugging image, add true and dir path at the end parameters.
    if(exist_img_right && set == kKITTI) {
      kitti_.Estimate3DLoc(param_, i, 
                           param_.save_image_disparity_, dir_disparity);
    }    
  }
  // Calculate fps.
  float fps = 1.0 / (duration_total / (float)size_dataset);
  std::cout << size_dataset << " images were processed (avg. "
    << std::setprecision(4) << fps << "fps)" << std::endl; 
  
  // If this function is called more than once, data structure must be cleared 
  // before each evaluation process. 
  data_all_.clear();
  // Evaluation - IoU.
  float iou = param_.eval_iou_start_;
  bool stay = true;
  const int iter_score = param_.eval_iter_score_;
  const int iter_score_low = param_.eval_iter_score_low_;
  while(stay) {
    // Condition for exit.
    if(fabs(param_.eval_iou_end_ - iou) <= std::numeric_limits<float>::epsilon()) {
      stay = false;
    }    
    
    int save_count = 0;
    std::vector<EvalDataPrecRec> data_subpart_prec_rec;
    // Varying score threshold, update precision and recall.
    float i = 0;
    while(i < iter_score) {
      const int i_wo_decimal = (int)i;
      float score_thres;
      if(iter_score == 0 || iter_score_low == 0) {
        ErrMsg(__func__, "iter_score(_low) should have positive number.");
        return false;
      }
      if(i >= iter_score - 1) {
        // No need to check (float)iter_score + (i-i_wo_decimal)
        // since iter_score is positive and (i-i_wo_decimal) is non-negative.
        score_thres = i_wo_decimal / (float)iter_score + (i-i_wo_decimal) / (float)iter_score_low;
      } else {
        score_thres = i_wo_decimal / (float)iter_score;
      }
      // Total Number of vehicles detected.
      unsigned int num_vehicle_pred = 0;
      // Total Number of true positive.
      unsigned int num_total_true_pos = 0;
      // Total Number of don't cares.
      unsigned int num_total_dont_care = 0;
      // Precision and recall.
      float precision = 0.;
      float recall = 0.;
      
      // For calculating precision and recall, and saving all locations.
      EvalDataPrecRec data_prec_rec;
      
      for(auto j = 0; j < size_dataset; ++j) {
        // Copy and remove detections under threshold.
        std::vector<Vehicle> detection;
        for(auto k = 0; k < ptr_data->at(j).detection_.size(); ++k) {
          if(ptr_data->at(j).detection_[k].GetScore() >= score_thres) {
            detection.push_back(ptr_data->at(j).detection_[k]);
          }
        }

        // Image saving part I - Visualize ground-truth bboxes and predicted bboxes.
        // 1. There's no meaningful bbox image for low and high threshold.
        // Too many bboxes are generated when the threshold is low.
        // Only small number of bboxes when the threshold is high.
        // Therefore, let's save the images with mid-range threshold only so that processing time can be saved.
        // 2. Also, similar score can be avoided.
        // 3. Varying IoU doesn't affect the detection itself, it only changes the performance measurement.
        // Therefore, we can save images only once for one iou iteration.
        if(param_.save_image_overlay_ == true && stay == false
           && score_thres >= param_.save_image_overlay_score_min_ 
           && score_thres <= param_.save_image_overlay_score_max_
           && save_count == 0) {
          // Make copy.
          cv::Mat img_save = ptr_data->at(j).img_[kCam_Left].clone();
          // Draw ground-truth bboxes first.
          VisualizeDetections(img_save, ptr_data->at(j).gt_vehicles_, kBbox_GT, kBbox_Score);
          // Draw dontcare bboxes.
          VisualizeDetections(img_save, ptr_data->at(j).gt_dontcare_, kBbox_dontcare, kBbox_Score); 
          // Draw predicted bboxes.
          VisualizeDetections(img_save, detection, kBbox_pred, kBbox_Score); 
          // Make directory.
          std::string file_name = ptr_data->at(j).file_name_;
          std::string dir_sub = dir_img + PathSeparator() 
            + file_name.substr(0, 4) + PathSeparator();
          mkdir(dir_sub.c_str(), 0777);
          // Append iteration number to file name.
          file_name.erase(file_name.end()-4, file_name.end());
          std::stringstream stream_score;
          stream_score << std::fixed << std::setprecision(2) << score_thres;
          file_name.append("_sco" + stream_score.str() + ".png");
          // Save image.        
          cv::imwrite(dir_sub + file_name, img_save);

          // Make another copy.
          cv::Mat img_coord = ptr_data->at(j).img_[kCam_Left].clone();
          // Draw predicted bboxes.
          VisualizeDetections(img_coord, ptr_data->at(j).gt_vehicles_, kBbox_GT, kBbox_Pos); 
          VisualizeDetections(img_coord, detection, kBbox_pred, kBbox_Pos); 
          file_name.erase(file_name.end()-4, file_name.end());
          file_name.append("_coord.png");
          // Save image.        
          cv::imwrite(dir_sub + file_name, img_coord);
        } 
        // End of Image saving part I.

        // Increase predicted vehicle count.
        num_vehicle_pred += detection.size();
        // Find True Positives and update results.
        // Assign file name as well.
        unsigned int num_true_pos, num_dont_care;
        if(!CountTP(ptr_data->at(j).gt_vehicles_,  
                    ptr_data->at(j).gt_dontcare_,
                    detection, data_prec_rec.data_loc_, 
                    num_true_pos, num_dont_care, iou,
                    ptr_data->at(j).file_name_)) {
          ErrMsg(__func__, "Error occured while counting TPs.");
          return false;
        }
        num_total_true_pos += num_true_pos; 
        num_total_dont_care += num_dont_care; 
      }
      
      // Calculate precision and recall.
      if (num_vehicle_pred == 0) {
        precision = 0.;
      } else {
        precision = num_total_true_pos / (float)(num_vehicle_pred - num_total_dont_care);
      }
      recall = num_total_true_pos / (float)num_vehicle_gt;  
      data_prec_rec.prec_ = precision;
      data_prec_rec.recall_ = recall;
      data_prec_rec.thres_ = score_thres;
      // Store precision and recall.
      data_subpart_prec_rec.push_back(data_prec_rec);

      if(++save_count == param_.save_image_overlay_interval_) {
         save_count = 0;
      }
      
      if(debug_msg) {
        std::cout << "IoU: " << iou;
        std::cout << ", ScoreThres: " << std::setprecision(2) << score_thres;
        std::cout << ", Precision: " << std::setprecision(4) << precision * 100
          << "% (" << num_total_true_pos << "/" 
          << num_vehicle_pred - num_total_dont_care << ")";
        std::cout << ", Recall: " << std::setprecision(4) << recall * 100
          << "% (" << num_total_true_pos << "/" << num_vehicle_gt << ")" << std::endl;
      }
      
      // To draw precision-recall graph well, we need the smaller division of the threshold in higher score.
      if(i >= iter_score - 1) {
        i = i + 1. / (float)iter_score_low;
      } else {
        i = i + 1;
      }      
    }
    
    // This contains mAP, IoU and data.
    // The data consists of precision and recall values.
    EvalDataMAP data_part_map;
  
    // When there're more than one precision value for one recall,
    // take the maximum precision value.
    // Sort first by ascending order then second by desceding order.
    sort(data_subpart_prec_rec.begin(), data_subpart_prec_rec.end(), SortByBoth);

    // Compute mAP.
    float mAP = 0.;
    for(int i = 0; i < data_subpart_prec_rec.size(); i++) {
      if(i == 0) {
        mAP += data_subpart_prec_rec[i].prec_ * data_subpart_prec_rec[i].recall_;
        data_part_map.prec_recall_.push_back(data_subpart_prec_rec[i]);
      // Find a point where recall(x) changes from the previous one.
      } else if (data_subpart_prec_rec[i].recall_ != data_subpart_prec_rec[i-1].recall_) {
        // First precision in the same recall is the highest, since the second elements were sorted by desceding order.
        mAP += (data_subpart_prec_rec[i].recall_ - data_subpart_prec_rec[i-1].recall_) * data_subpart_prec_rec[i].prec_;
        data_part_map.prec_recall_.push_back(data_subpart_prec_rec[i]);         
      }    
    }
    // Update and store all data.
    data_part_map.iou_ = iou;
    data_part_map.mAP_ = mAP;
    std::cout << "IoU: " << iou << " mAP: " << mAP << std::endl;
    data_all_.push_back(data_part_map);
    
    // Update IoU.
    iou += param_.eval_iou_step_;
  }
  
  std::cout << "Evaluation completed for " << size_dataset << " images." << std::endl;

  return true;
}
  
// Count true positive from ground-truth vehicles and predicted vehicles.
bool PerformanceEval::CountTP(std::vector<Vehicle>& gt, 
                             std::vector<Vehicle>& dontcare,
                             std::vector<Vehicle>& pred, 
                             std::vector<EvalDataLoc>& data_loc,
                             unsigned int& num_true_pos,
                             unsigned int& num_dont_care,
                             const float overlap_thres,
                             const std::string& file_name) {
  num_true_pos = 0;
  num_dont_care = 0;
  bool match = false;
  
  // Save flags when any of bbox is matched with the prediction.
  std::vector<bool> match_gt(gt.size(), false);
  std::vector<bool> match_dc(dontcare.size(), false);
  // Index matched.
  int match_index;
  
  // Find match for each prediction.
  for(auto it_pred = pred.begin(); it_pred != pred.end(); ++it_pred) {
    int pred_idx = std::distance(pred.begin(), it_pred);
    float max = overlap_thres;
    match = false;
    EvalDataLoc data_loc_part;
    data_loc_part.file_name_ = file_name;
      
    cv::Rect bbox_pred = pred[pred_idx].GetBbox();
    // Exception.
    if(bbox_pred.width <= 0. || bbox_pred.height <= 0.) {
      std::cerr << __func__ << ": Invalid bbox."
      << "[Pred]w:" << bbox_pred.width << " h:" << bbox_pred.height << std::endl;
      continue;
    }
    
    // Search GT first.
    for(auto it_gt = gt.begin(); it_gt != gt.end(); ++it_gt) {
      int idx = std::distance(gt.begin(), it_gt);
      // Skip if any bbox is already matched before.
      if(match_gt[idx] == false) {
        const float iou = CalculateIoU(bbox_pred, (*it_gt).GetBbox());
        // Take the maximum score and its index.
        if(iou >= max) {
          max = iou;
          match = true;
          match_index = idx;
        }
      } 
    }
  
    // Update.
    if (match == true) {
      match_gt[match_index] = true;
      // Increase true positive number.
      num_true_pos++;
      // Save 3d location.
      // If it's matched with GT-Bbox, save GT-Bbox location.
      data_loc_part.loc_3D_[k3D_X] = gt[match_index].Get3DLoc(k3D_X);
      data_loc_part.loc_3D_[k3D_Y] = gt[match_index].Get3DLoc(k3D_Y);
      data_loc_part.loc_3D_[k3D_Z] = gt[match_index].Get3DLoc(k3D_Z);
      data_loc_part.result_ = kTruePos;
      data_loc_part.bbox_ = gt[match_index].GetBbox();
      data_loc_part.difficulty_ = gt[match_index].GetDifficulty();
      data_loc.push_back(data_loc_part);
    } else {
      // Search Dontcare.
      if(dontcare.size() > 0) {
        // Search dontcare only when there's no match with gt.
        for(auto it_dc = dontcare.begin(); it_dc != dontcare.end(); ++it_dc) {
          int idx = std::distance(dontcare.begin(), it_dc);
          // Skip if any bbox is already matched before.
          if(match_dc[idx] == false) {
            const float iou = CalculateIoU(bbox_pred, (*it_dc).GetBbox(),
                                       kBbox_dontcare);
            if(iou >= max) {
              max = iou;
              match = true;
              match_index = idx;
            }
          }
        }
      }
      
      // Update - True positive
      if (match == true) {
        match_dc[match_index] = true;
        // Increase don't care number.
        num_dont_care++;
        data_loc_part.result_ = kDontCare;
      // NOT matched with neither gt nor dontcare - add False positives.
      } else {
        data_loc_part.result_ = kFalsePos;
        data_loc_part.bbox_ = bbox_pred;
      }
      // Save 3d location.
      data_loc_part.loc_3D_[k3D_X] = pred[pred_idx].Get3DLoc(k3D_X);
      data_loc_part.loc_3D_[k3D_Y] = pred[pred_idx].Get3DLoc(k3D_Y);
      data_loc_part.loc_3D_[k3D_Z] = pred[pred_idx].Get3DLoc(k3D_Z);
      data_loc_part.difficulty_ = kDifficulty_Unknown;
      data_loc.push_back(data_loc_part);
    } 
  }
  
  // Find unmatched GTs.
  for(auto it_gt = gt.begin(); it_gt != gt.end(); ++it_gt) {
    int idx = std::distance(gt.begin(), it_gt);
    // Skip if any bbox is already matched before.
    if(match_gt[idx] == false) {
      // Save 3d location.
      EvalDataLoc data_loc_part;
      data_loc_part.file_name_ = file_name;
      data_loc_part.loc_3D_[k3D_X] = gt[idx].Get3DLoc(k3D_X);
      data_loc_part.loc_3D_[k3D_Y] = gt[idx].Get3DLoc(k3D_Y);
      data_loc_part.loc_3D_[k3D_Z] = gt[idx].Get3DLoc(k3D_Z);
      data_loc_part.result_ = kFalseNeg;
      data_loc_part.bbox_ = gt[idx].GetBbox();
      data_loc_part.difficulty_ = gt[idx].GetDifficulty();
      data_loc.push_back(data_loc_part);
    }
  }

  return true;
}

// Save TP, FP and FN images
bool PerformanceEval::SaveBbox(const unsigned int set, 
                               const int idx_iou, const int idx_thres) {
  // Check flag.
  if(!param_.save_image_bbox_) {
    return true;
  }
  
  // If parameter indices are valid, use them.
  // Otherwise, find the indices.
  int idx1 = -1;
  int idx2 = -1;
  if (idx_iou >= 0 && idx_thres >= 0) { 
    idx1 = idx_iou;
    idx2 = idx_thres;
  } else {
    // Find IoU.
    for(auto it = data_all_.begin(); it != data_all_.end(); ++it) {
      const int idx = std::distance(data_all_.begin(), it);
      // Search for desired IoU.
      if (fabs(data_all_[idx].iou_ - param_.save_image_bbox_iou_) 
          <= std::numeric_limits<float>::epsilon()) {
        idx1 = idx;
        break;
      }
    }
    if(idx1 < 0) {
      ErrMsg(__func__, "IoU value not found");
      return false;
    }
    // Find score.
    for(auto it = data_all_[idx1].prec_recall_.begin(); 
        it != data_all_[idx1].prec_recall_.end(); ++it) {
      const int idx = std::distance(data_all_[idx1].prec_recall_.begin(), it);
      if (fabs(data_all_[idx1].prec_recall_[idx].thres_ - param_.save_image_bbox_score_) 
          <= std::numeric_limits<float>::epsilon()) {
        idx2 = idx;
        break;
      }
    }
    if(idx2 < 0) {
      ErrMsg(__func__, "Threshold(score) value not found");
      std::cerr << "Score: " << param_.save_image_bbox_score_ << std::endl;
      return false;
    }
  }     
  std::cout << "Saving bbox for iou " 
    << data_all_[idx1].iou_ << " and threshold " 
    << data_all_[idx1].prec_recall_[idx2].thres_ << std::endl;
    
  std::vector<std::vector<int>> midx(kRESULT_num, std::vector<int>(0,0));
  // Search results.
  for(auto it_pred = data_all_[idx1].prec_recall_[idx2].data_loc_.begin();
      it_pred != data_all_[idx1].prec_recall_[idx2].data_loc_.end(); ++it_pred) {
    int i = std::distance(data_all_[idx1].prec_recall_[idx2].data_loc_.begin(), it_pred);
    midx[data_all_[idx1].prec_recall_[idx2].data_loc_[i].result_].push_back(i);
  }
                    
  // For each class (TP, FN and FP)
  int idx_start = 0;
  if(!param_.save_image_bbox_tp_) {
    idx_start++;  
  }
  for(int idx_result = idx_start; idx_result < kRESULT_num - 1; ++idx_result) {
    // Create result dirs.
    const std::string dir_make = param_.dir_result_ + PathSeparator()
      + param_.save_image_bbox_dir_[idx_result] + PathSeparator(); 
    mkdir(dir_make.c_str(), 0777);
    
    if(midx[idx_result].size() > 0) {
      // Initialize iterators.
      int k = 0;
      int idx = midx[idx_result][k];
      while (k < midx[idx_result].size()) {
        // The value "idx" always comes from the index of data_loc.
        // No need to reference check here, but just in case.
        if(idx < 0 || idx > data_all_[idx1].prec_recall_[idx2].data_loc_.size()) {
          ErrMsg(__func__, "Index is invalid.");
          return false;
        }
        // Load image.
        const std::string dir_name = param_.dir_dataset_test_image_[set] + PathSeparator();
        const std::string file_name = data_all_[idx1].prec_recall_[idx2].data_loc_[idx].file_name_;
        cv::Mat img = cv::imread(dir_name + file_name, -1);

        int cnt = 0;
        // Match file name.
        while (k < midx[idx_result].size() 
               && !data_all_[idx1].prec_recall_[idx2].data_loc_[idx].file_name_.compare(file_name)) {
          // Get image cropped.
          cv::Mat crop = img(data_all_[idx1].prec_recall_[idx2].data_loc_[idx].bbox_);
          // Remove extension from the filename.
          std::string file_wo_ext = file_name;
          file_wo_ext.erase(file_wo_ext.end()-4, file_wo_ext.end());
          // Distance.
          const float dist = data_all_[idx1].prec_recall_[idx2].data_loc_[idx].GetDist();
          const int diff = data_all_[idx1].prec_recall_[idx2].data_loc_[idx].difficulty_;
          // Append number to the file name and save image.
          cv::imwrite(dir_make + file_wo_ext + "_" + std::to_string(cnt++) 
                      + "_" + std::to_string(dist) 
                      + "_" + std::to_string(diff) + ".jpg", crop);

          idx = midx[idx_result][++k];
        }
      }  
    }
  }

  return true;
}  
  
// Draw background of line plot.
void PerformanceEval::VisualizePlotLineBG(cv::Mat img) {
  // Image size.
  const unsigned int size_w = plot_line_size_w_;
  const unsigned int size_h = plot_line_size_h_;
  const unsigned int in_size = plot_line_size_in_;
  const unsigned int margin = (size_h - in_size) * 0.5;
  // For text.
  const int fontFace = cv::FONT_HERSHEY_PLAIN;
  const double fontScale = font_size_;
  const double thickness = font_thickness_;
  const unsigned int txt_offset = font_space_;
  int offset_x = 0;
  int offset_y = 0;
 
  // Paint background of the graph.
  cv::rectangle(img, cv::Point(margin, margin), cv::Point(margin + in_size, margin + in_size),
           inner_bg_color_, CV_FILLED);

  // Draw grid.
  for(int i = 0; i <= in_size; i = i + in_size / 10) {
    // Vertical lines.
    cv::line(img, cv::Point(margin + i, margin), cv::Point(margin + i, margin + in_size),
             grid_line_color_);
    // Horizontal lines.
    cv::line(img, cv::Point(margin, margin + i),
             cv::Point(margin + in_size, margin + i), grid_line_color_);
    // Text.
    std::stringstream stream;
    stream << std::fixed << std::setprecision(1) << i/(float)in_size;
    std::string text = stream.str();
    // X-axis.
    if (i != 0) {
      offset_x = plot_line_pos_x_[0] * txt_offset;
      offset_y = plot_line_pos_x_[1] * txt_offset;
      cv::putText(img, text,
                  cv::Point(margin + i + offset_x, margin + in_size + offset_y),
                  fontFace, fontScale, cv::Scalar::all(0), thickness, 8);
    }
    // Y-axis.
    offset_x = plot_line_pos_y_[0] * txt_offset;
    offset_y = plot_line_pos_y_[1] * txt_offset;
    cv::putText(img, text, cv::Point(margin + offset_x, size_h - margin - i + offset_y),
                fontFace, fontScale, cv::Scalar::all(0), thickness, 8);
  }

  // Write axis - recall and precision.
  offset_x = plot_line_pos_prec_[0] * txt_offset;
  offset_y = plot_line_pos_prec_[1] * txt_offset;
  cv::putText(img, "Prec", cv::Point(offset_x, size_h * 0.5 + offset_y), fontFace, fontScale, 
              cv::Scalar::all(0), thickness, 8);
  offset_x = plot_line_pos_recall_[0] * txt_offset;
  offset_y = plot_line_pos_recall_[1] * txt_offset;
  cv::putText(img, "Recall", cv::Point(size_h * 0.5 + offset_x, margin + in_size + offset_y), 
              fontFace, fontScale, cv::Scalar::all(0), thickness, 8); 
  // Title of the graph.
  offset_x = plot_line_pos_title_[0] * txt_offset;
  offset_y = plot_line_pos_title_[1] * txt_offset;
  cv::putText(img, "Precision-Recall curve", cv::Point(size_h * 0.5 + offset_x, margin + offset_y), fontFace, fontScale * 1.6, 
              cv::Scalar::all(0), thickness * 1.2, 8);
}
  
// Draw and save evaluation result. 
void PerformanceEval::VisualizePlotLineResult() {
  // Image size.
  const unsigned int size_w = plot_line_size_w_;
  const unsigned int size_h = plot_line_size_h_;
  const unsigned int in_size = plot_line_size_in_;
  const unsigned int margin = (size_h - in_size) * 0.5;
  // For text.
  const int fontFace = cv::FONT_HERSHEY_PLAIN;
  const double fontScale = font_size_;
  const double thickness = font_thickness_;
  const unsigned int txt_offset = font_space_;
  int offset_x = 0;
  int offset_y = 0;

  // Create an image.
  cv::Mat recall_prec_graph(size_h, size_w, CV_8UC3, bg_color_);

  // Draw background.
  VisualizePlotLineBG(recall_prec_graph); 
 
  // Draw and Write results.
  const unsigned int result_x = margin + in_size + txt_offset;
  const unsigned int result_y = margin + txt_offset;
  // Box
  const unsigned int space_row = plot_txt_space_row_;
  cv::rectangle(recall_prec_graph, cv::Point(result_x, result_y),
                cv::Point(size_w - txt_offset, result_y + (data_all_.size() + 1) * space_row + txt_offset),
                inner_bg_color_);
  // Entity.
  cv::putText(recall_prec_graph, "    IoU   mAP", 
              cv::Point(result_x + txt_offset, result_y + space_row),
              fontFace, fontScale, cv::Scalar::all(0), thickness, 8);
  
  for(int i = 0; i < data_all_.size(); ++i) {    
    // IoU and mAP Values.
    std::stringstream stream_mAP, stream_iou;
    stream_mAP << std::fixed << std::setprecision(3) << data_all_[i].mAP_;
    stream_iou << std::fixed << std::setprecision(2) << data_all_[i].iou_;
    std::string text_res = "   " + stream_iou.str() + "  " + stream_mAP.str();
    offset_x = plot_line_pos_ap_[0] * txt_offset;
    offset_y = plot_line_pos_ap_[1] * txt_offset;
    cv::putText(recall_prec_graph, text_res, cv::Point(result_x + txt_offset, result_y + space_row * (i+2)),
                fontFace, fontScale, cv::Scalar::all(0), thickness, 8);
    // Color index.
    cv::circle(recall_prec_graph, cv::Point(result_x + txt_offset * 3, result_y + space_row * (i+2) - txt_offset),
               3, param_.eval_line_color_[i], 5);

    // Draw line between two points.
    if(data_all_[i].prec_recall_.size() > 1) {
      for(int j = 1; j < data_all_[i].prec_recall_.size(); ++j) {
        float recall_prev = data_all_[i].prec_recall_[j-1].recall_ * in_size + margin;
        float precision_prev = (1.0 - data_all_[i].prec_recall_[j-1].prec_) * in_size + margin;
        float recall = data_all_[i].prec_recall_[j].recall_ * in_size + margin;
        float precision = (1.0 - data_all_[i].prec_recall_[j].prec_) * in_size + margin;
        // Draw line.
        cv::line(recall_prec_graph, 
                cv::Point(recall_prev, precision_prev),
                cv::Point(recall, precision), param_.eval_line_color_[i], 2);      
      }
    }
  } 
  
  // Create default result directory.
  mkdir(param_.dir_result_.c_str(), 0777);
  
  // Save image: stored in result directory. 
  const std::string file = param_.dir_result_ + PathSeparator() + "Result_Precision_Recall.jpg";
  cv::imwrite(file, recall_prec_graph);
}

// Draw background of radius plot.
void PerformanceEval::VisualizePlotRadBG(cv::Mat img, const unsigned int max_dist) {
  // Image size - width, height, origin of the camera (car).
  const unsigned int size_w = plot_sct_size_w_;
  const unsigned int size_h = plot_sct_size_h_;
  const unsigned int origin_x = (int)(size_w / 2);
  const unsigned int origin_y = size_h - plot_sct_margin_btm_;
  // For text print.
  const int fontFace = cv::FONT_HERSHEY_PLAIN;
  const double fontScale = font_size_;
  const double thickness = font_thickness_;
  const unsigned int space_row = plot_txt_space_row_;
  // Field of View and scale of the plotting (meter to pixel).
  const float fov_half = param_.eval_cam_fov_ / 2;
  const unsigned int scale = plot_sct_scale_ * plot_sct_dist_max_ / max_dist;
  const int angle = 0;
  
  // Draw arc - paint.
  cv::Scalar color_far = CV_RGB(plot_sct_color_bg_.val[0] + 15,
                                plot_sct_color_bg_.val[1] + 15,
                                plot_sct_color_bg_.val[2] + 15);
  cv::ellipse(img, cv::Point(origin_x, origin_y),
              cv::Size(plot_sct_dist_max_ * scale * 2, plot_sct_dist_max_ * scale * 2),
              0, 270.0 - fov_half, 270.0 + fov_half,
              color_far, CV_FILLED);
  cv::ellipse(img, cv::Point(origin_x, origin_y),
              cv::Size(plot_sct_dist_near_ * scale, plot_sct_dist_near_ * scale),
              0, 270.0 - fov_half, 270.0 + fov_half,
              plot_sct_color_bg_, CV_FILLED);
  // Draw arc for every like 10m.
  for(int i = 0; i <= plot_sct_dist_max_; i+=plot_sct_dist_interval_) {
    cv::Scalar line_color;
    if(i % 10 == 0) {
      // Set bold line.
      line_color = plot_sct_color_line_thick_;
      
      if (i != 0) {
        // Write distance.
        std::string meter = std::to_string(i);
        meter += "m";
        const unsigned int pos_x = origin_x 
          + cos(DegToRad(90 - fov_half)) * i * scale + plot_sct_pos_unit_[0];
        const unsigned int pos_y = origin_y 
          - sin(DegToRad(90 - fov_half)) * i * scale + plot_sct_pos_unit_[1];
        cv::putText(img, meter, cv::Point(pos_x, pos_y),
                    fontFace, fontScale, cv::Scalar::all(0), thickness, 8);
      }
    } else {
      // Set light line.
      line_color = plot_sct_color_line_thin_;
    }  
    
    // Draw lines.
    cv::ellipse(img, cv::Point(origin_x, origin_y),
              cv::Size(i * scale, i * scale),
              0, 270.0 - fov_half, 270.0 + fov_half, line_color);
  }
  // Draw blod line at the border of near-far region.
  cv::ellipse(img, cv::Point(origin_x, origin_y),
              cv::Size(plot_sct_dist_near_ * scale, plot_sct_dist_near_ * scale),
              0, 270.0 - fov_half, 270.0 + fov_half, plot_sct_color_line_thick_, 2);
  
  // Draw axis.
  const unsigned int len_axis = plot_sct_length_axis_;
  // Draw Z-axis.
  // Center line.
  cv::line(img, cv::Point(origin_x, 10), cv::Point(origin_x, origin_y),
           100);
  // Bold arrow.
  cv::line(img, cv::Point(origin_x, origin_y), 
           cv::Point(origin_x, origin_y - len_axis),
           plot_sct_color_line_thick_, 2);
  cv::line(img, cv::Point(origin_x, origin_y - len_axis), 
           cv::Point(origin_x + 5, origin_y - len_axis + 5),
           plot_sct_color_line_thick_, 2);
  cv::line(img, cv::Point(origin_x, origin_y - len_axis), 
           cv::Point(origin_x - 5, origin_y - len_axis + 5),
           plot_sct_color_line_thick_, 2);
  // +Z Text.
  cv::putText(img, "+Z", 
              cv::Point(origin_x + plot_sct_pos_z_[0], origin_y - len_axis + plot_sct_pos_z_[1]),
              fontFace, fontScale, cv::Scalar::all(0), thickness, 8);
  // Draw X-axis.
  // Bold arrow.
  cv::line(img, cv::Point(origin_x, origin_y), 
           cv::Point(origin_x + len_axis, origin_y),
           plot_sct_color_line_thick_, 2);
  cv::line(img, cv::Point(origin_x + len_axis, origin_y), 
           cv::Point(origin_x + len_axis - 5, origin_y - 5),
           plot_sct_color_line_thick_, 2);
  cv::line(img, cv::Point(origin_x + len_axis, origin_y), 
           cv::Point(origin_x + len_axis - 5, origin_y + 5),
           plot_sct_color_line_thick_, 2);
  // +X Text.
  cv::putText(img, "+X", 
              cv::Point(origin_x + len_axis + plot_sct_pos_x_[0], origin_y + plot_sct_pos_x_[1]),
              fontFace, fontScale, cv::Scalar::all(0), thickness, 8);
}
  
// Visualize detection results over camera radius.
void PerformanceEval::VisualizePlotRadResult(int& idx_iou, int& idx_thres) {
  // Check flag.
  if(!param_.save_image_birdeye_) {
    return;
  }

  // Create default result directory.
  mkdir(param_.dir_result_.c_str(), 0777);
  
  // Check directory and create if required.
  DIR *dir_birdeye;
  const std::string birdeye_dir = param_.dir_result_ + PathSeparator() + param_.save_image_birdeye_dir_;
  if(!OpenDirectory(birdeye_dir, &dir_birdeye, __func__)) {
    std::cout << "Creating directory: " << birdeye_dir << std::endl;
    mkdir(birdeye_dir.c_str(), 0777);    
  } else {
    closedir(dir_birdeye);
  }
  
  // Image size - width, height, origin of the camera (car).
  const unsigned int size_w = plot_sct_size_w_;
  const unsigned int size_h = plot_sct_size_h_;
  const unsigned int origin_x = (int)(size_w / 2);
  const unsigned int origin_y = size_h - plot_sct_margin_btm_;
  // For text print.
  const int fontFace = cv::FONT_HERSHEY_PLAIN;
  const double fontScale = font_size_;
  const double thickness = font_thickness_;
  const unsigned int space_row = plot_txt_space_row_;
  // Field of View and scale of the plotting (meter to pixel).
  const float fov_half = param_.eval_cam_fov_ / 2;
  const unsigned int scale = plot_sct_scale_;
  const int angle = 0;
  
  // Create an white background image.
  cv::Mat radius_template(size_h, size_w, CV_8UC3, cv::Scalar::all(255));
  cv::Mat radius_template_zoom(size_h, size_w, CV_8UC3, cv::Scalar::all(255));
  
  // Draw background - entire view and zoomed view.
  VisualizePlotRadBG(radius_template, plot_sct_dist_max_);
  VisualizePlotRadBG(radius_template_zoom, plot_sct_dist_near_);
  
  // Get configuration values - IoU and Target precision and recall.
  const float iou = param_.save_image_birdeye_iou_;
  const float prec_target = param_.save_image_birdeye_prec_target_;
  const float prec_w = param_.save_image_birdeye_prec_weight_;
  const float recall_target = param_.save_image_birdeye_recall_target_;
  const float recall_w = param_.save_image_birdeye_recall_weight_;
  
  // Find IoU.
  int idx1;
  for(auto it_1 = data_all_.begin(); it_1 != data_all_.end(); ++it_1) {
    const int idx = std::distance(data_all_.begin(), it_1);
    
    // Search for desired IoU.
    if (fabs(data_all_[idx].iou_ - iou) <= std::numeric_limits<float>::epsilon()) {
      idx1 = idx;
      break;
    }
  }
  
  // Find precision and recall.
  int idx2;
  bool beyond_target = false;
  float dist_max = -1.;
  float dist_min = prec_w + recall_w; // big enough than the the max distance within the prec-recall graph.
  for(auto it_2 = data_all_[idx1].prec_recall_.begin(); 
      it_2 != data_all_[idx1].prec_recall_.end(); ++it_2) {
    const int idx = std::distance(data_all_[idx1].prec_recall_.begin(), it_2);

    // Get precision and recall.
    const float prec = data_all_[idx1].prec_recall_[idx].prec_;
    const float recall = data_all_[idx1].prec_recall_[idx].recall_;
    // Calculate distance from the target by applying weights.
    const float dist_prec = (prec_target - prec) * prec_w;
    const float dist_recall = (recall_target - recall) * recall_w;
    const float dist = sqrt(dist_prec*dist_prec + dist_recall*dist_recall);
    
    // Check if the target is already reached.
    if(prec >= prec_target && recall >= recall_target) {
      beyond_target = true;
      // Update with the farthest point.
      if(dist > dist_max) {
        dist_max = dist;
        idx2 = idx;
      }
    } else if(beyond_target == false) {
      // Update with the closest point.
      if(dist < dist_min) {
        dist_min = dist;
        idx2 = idx;
      }
    }
  }
               
  // Create result image.
  cv::Mat radius_result = radius_template.clone();
  cv::Mat radius_result_sub = radius_template.clone();
  cv::Mat radius_result_zoom = radius_template_zoom.clone();
  cv::Mat radius_result_fail_zoom = radius_template_zoom.clone();
          
  unsigned int count_TP[kNum_Dist] = {0, 0};
  unsigned int count_FN[kNum_Dist] = {0, 0};
  unsigned int count_FP[kNum_Dist] = {0, 0};
        
  // Sort by file name.
  // Note, this is not mendatory because the same file name appears continuously.
  sort(data_all_[idx1].prec_recall_[idx2].data_loc_.begin(),
       data_all_[idx1].prec_recall_[idx2].data_loc_.end(),
       [](EvalDataLoc const& v1, EvalDataLoc const& v2) {
   return v1.file_name_ < v2.file_name_;} );
        
  // Write information on string - iou, threshold, precision, recall.
  std::stringstream str_iou, str_thres, str_prec, str_rec;
  str_iou << std::fixed << std::setprecision(2) << iou;
  str_thres << std::fixed << std::setprecision(3)
    << data_all_[idx1].prec_recall_[idx2].thres_;
  str_prec << std::fixed << std::setprecision(4)
    << data_all_[idx1].prec_recall_[idx2].prec_;
  str_rec << std::fixed << std::setprecision(4)
    << data_all_[idx1].prec_recall_[idx2].recall_; 
  // Generate file name.
  const std::string file_common = birdeye_dir + PathSeparator() 
    + "Birdeye_iou" + str_iou.str() + "_thres" + str_thres.str()
    + "_prec" + str_prec.str();
    
  // Draw circles.
  std::string file_name = "";
  unsigned int file_count = 0;
  unsigned int image_count = 0;
  for(auto it_3 = data_all_[idx1].prec_recall_[idx2].data_loc_.begin();
      it_3 != data_all_[idx1].prec_recall_[idx2].data_loc_.end();
      ++it_3) {
    int idx3 = std::distance(data_all_[idx1].prec_recall_[idx2].data_loc_.begin(), it_3);
    // Get data.
    EvalDataLoc data = data_all_[idx1].prec_recall_[idx2].data_loc_[idx3];

    // Check file name and count number of files.
    bool write = false;
    if(idx3 == data_all_[idx1].prec_recall_[idx2].data_loc_.size()-1) {
      write = true;
    }

    if(file_name.compare(data.file_name_) || write) {
      file_name = data.file_name_;
      file_count++;

      // Reset count.
      if(file_count > param_.save_image_birdeye_num_subsamples_ || write) {
        file_count = 1;
        image_count++;

        // Generate file name.
        const std::string file_sub = file_common + "_rec" + str_rec.str() 
          + "_part" + std::to_string(image_count) + ".jpg";
        // Save image.
        cv::imwrite(file_sub, radius_result_sub);
        // Refresh image.
        radius_result_sub = radius_template.clone();        
      }                            
    }
          
    const unsigned int pos_x = origin_x + data.loc_3D_[k3D_X] * scale;
    const unsigned int pos_y = origin_y - data.loc_3D_[k3D_Z] * scale;
    const unsigned int pos_zoom_x = origin_x 
      + data.loc_3D_[k3D_X] * scale * plot_sct_dist_max_ / plot_sct_dist_near_;
    const unsigned int pos_zoom_y = origin_y
      - data.loc_3D_[k3D_Z] * scale * plot_sct_dist_max_ / plot_sct_dist_near_;

    // Choose area depending on the distance.
    unsigned int area = -1;
    if(data.GetDist() <= plot_sct_dist_near_) {
      area = kNear;
    } else {
      area = kFar;
    }
    // Count and set color.
    cv::Scalar circle_color;
    switch (data.result_) {
      case kTruePos:
        count_TP[area]++;
        circle_color = plot_sct_color_TP_;
        break;
      case kFalseNeg:
        count_FN[area]++;
        circle_color = plot_sct_color_FN_;
        break;
      case kFalsePos:
        count_FP[area]++;
        circle_color = plot_sct_color_FP_;
        break;
      case kDontCare:
        circle_color = plot_sct_color_DC_;
        break;
    }

    // Draw the vehicle point.
    cv::circle(radius_result, cv::Point(pos_x, pos_y), 2, circle_color, 2);
    cv::circle(radius_result_sub, cv::Point(pos_x, pos_y), 2, circle_color, 2);
    cv::circle(radius_result_zoom, cv::Point(pos_zoom_x, pos_zoom_y), 2,
               circle_color, 2);
    if(data.result_ != kTruePos) {
      cv::circle(radius_result_fail_zoom, cv::Point(pos_zoom_x, pos_zoom_y), 2,
                 circle_color, 2);
      // Write file name.
      // For the KITTI, file name is trimmed to make it short.
      std::string file = data.file_name_.substr(2,4);
      cv::putText(radius_result_fail_zoom, file, 
      cv::Point(pos_zoom_x+2, pos_zoom_y-2), fontFace, fontScale/1.3, cv::Scalar::all(0), thickness, 8);
    }
  }
      
  // Draw in the image.
  unsigned int pos_x = plot_sct_pos_info1_[0];
  unsigned int pos_y = size_h + plot_sct_pos_info1_[1];        
  cv::putText(radius_result, "Precision: " + str_prec.str(), 
      cv::Point(pos_x, pos_y), fontFace, fontScale, cv::Scalar::all(0), thickness, 8);
  pos_y -= plot_txt_space_row_;
  cv::putText(radius_result, "Recall: " + str_rec.str(), 
      cv::Point(pos_x, pos_y), fontFace, fontScale, cv::Scalar::all(0), thickness, 8);        
  pos_y -= plot_txt_space_row_;
  cv::putText(radius_result, "Score threshold: " + str_thres.str(), 
      cv::Point(pos_x, pos_y), fontFace, fontScale, cv::Scalar::all(0), thickness, 8);        
  pos_y -= plot_txt_space_row_;
  cv::putText(radius_result, "IoU: " + str_iou.str(), 
      cv::Point(pos_x, pos_y), fontFace, fontScale, cv::Scalar::all(0), thickness, 8);        

  // Write information on string - FOV.
  std::stringstream str_fov;
  str_fov << std::fixed << std::setprecision(2) << param_.eval_cam_fov_;

  // Draw in the image.
  pos_x = size_w + plot_sct_pos_info2_[0];
  pos_y = size_h + plot_sct_pos_info2_[1];  
  cv::putText(radius_result, "Cam Hor. FOV: " + str_fov.str(), 
      cv::Point(pos_x, pos_y), fontFace, fontScale, cv::Scalar::all(0), thickness, 8);

  // Calculate recalls and precisions, and write on string.
  float recall[kNum_Dist] = {0., 0.};
  float prec[kNum_Dist] = {0., 0.};
  std::stringstream str_recall_area[kNum_Dist], str_prec_area[kNum_Dist];
  std::string recall_str[kNum_Dist], prec_str[kNum_Dist];
        
  for (int i = 0; i < kNum_Dist; ++i) {
    // Calculate Recall and Precision.
    if (count_TP[i] + count_FN[i] != 0) {
      recall[i] = count_TP[i] / (float)(count_TP[i] + count_FN[i]);
      str_recall_area[i] << std::fixed << std::setprecision(3) << recall[i];
    }
    if (count_TP[i] + count_FP[i] != 0) {
      prec[i] = count_TP[i] / (float)(count_TP[i] + count_FP[i]);
      str_prec_area[i] << std::fixed << std::setprecision(3) << prec[i];
    }

    // Generate output string.
    const std::string str_dist = distance_str[i];
    recall_str[i] = "Recall (" + str_dist
      + "): " + str_recall_area[i].str()
      + " (" + std::to_string(count_TP[i])
      + "/" + std::to_string(count_TP[i]+count_FN[i]) + ")";
    prec_str[i] = "Precision (" + str_dist
      + "): " + str_prec_area[i].str()
      + " (" + std::to_string(count_TP[i])
      + "/" + std::to_string(count_TP[i]+count_FP[i]) + ")";
  }

  // Draw in the image.
  pos_y -= plot_txt_space_row_;
  cv::putText(radius_result, prec_str[kFar], 
      cv::Point(pos_x, pos_y), fontFace, fontScale, cv::Scalar::all(0), thickness, 8);
  pos_y -= plot_txt_space_row_;
  cv::putText(radius_result, prec_str[kNear], 
      cv::Point(pos_x, pos_y), fontFace, fontScale, cv::Scalar::all(0), thickness, 8);
  pos_y -= plot_txt_space_row_;
  cv::putText(radius_result, recall_str[kFar], 
      cv::Point(pos_x, pos_y), fontFace, fontScale, cv::Scalar::all(0), thickness, 8);
  pos_y -= plot_txt_space_row_;
  cv::putText(radius_result, recall_str[kNear], 
      cv::Point(pos_x, pos_y), fontFace, fontScale, cv::Scalar::all(0), thickness, 8);

  // Generate file name.
  const std::string file_all = file_common + "(n" + str_prec_area[kNear].str() 
    + "f" + str_prec_area[kFar].str() + ")_rec" + str_rec.str() + "(n" 
    + str_recall_area[kNear].str() + "f" + str_recall_area[kFar].str() + ")";
  const std::string file = file_all + ".jpg";
  const std::string file_zoom = file_all + "_zoom.jpg";
  const std::string file_failzoom = file_all + "_zoom_fail.jpg";

  // Save image: stored in script directory.        
  cv::imwrite(file, radius_result);
  cv::imwrite(file_zoom, radius_result_zoom);
  cv::imwrite(file_failzoom, radius_result_fail_zoom);

  // Save indices for future use.
  idx_iou = idx1;
  idx_thres = idx2;
}
  
// Load models and all parameters needed to perform evaluation.
bool PerformanceEval::LoadAll(const std::string& model_name,
                              const std::string& trained_file) {
  // Load model.
  if(!LoadModelByName(model_name, trained_file)) {
    return false;
  }
  std::cout << "LoadModel() complete" << std::endl;
  
  // Load plot parameters.
  std::cout << "Load plot file (" << param_.file_plotting_ << ") ...";
  if(!LoadParamEvalPlot(param_.file_plotting_)) {
    ErrMsg(__func__, "failed.");
    return false;
  }
  std::cout << "complete" << std::endl;
  
  return true;
}
  
  
} // namespace vehicle_detector
} // namespace robotics

