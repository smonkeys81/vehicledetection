// Copyright © 2020 Robotics, Inc. All Rights Reserved.

// This file defines components of dataset and loading method.

#include "dataset.h"


namespace robotics {
namespace vehicle_detector {


// Estimate 3d location of the prediction by using stereo images.
bool DataSet::Estimate3DLoc(const Param param,
                            const unsigned int idx,
                            const bool debug,
                            const std::string& save_dir) {
  // There are two options available to estimate 3D location.
  // 1. Extract features from bbox, make descriptors, find correspondence,
  // and calculate depth.
  // 2. Make disparity map for entire image by using opencv, and calculate depth.
  // First, we use method 2 because we can expect more dense disparity, while method 1
  // may have no feature in the bbox, particularly when the bbox is small.
  // Cons of the method 2 is that it requires more computation.
  // So far, the computation time is not critical since we use this for only evaluation.
  // (Currently, the average processing time is 15ms for each pair of images.)
  // If we want to estimate 3d location while detecting, we will have to consider
  // this method again.
  
  // Create disparity image (16bit signed).
  cv::Mat img_disparity16S = cv::Mat(dataset_[idx].img_[kCam_Left].rows,
                                     dataset_[idx].img_[kCam_Left].cols,
                                     CV_16S);
  // Initialize with 0.
  // Otherwise, garbage value remains when calculating the average.
  img_disparity16S = cv::Scalar::all(0);

  // Create normalized disparity image.
  cv::Mat img_disparity8 = cv::Mat(dataset_[idx].img_[kCam_Left].rows,
                                   dataset_[idx].img_[kCam_Left].cols,
                                   CV_8U);

  // The disparity image (CS_16S) is scaled by 16 - not configurable.
  const unsigned int disp_scale = 16;
  
  // Create gray images and convert from the color images.
  cv::Mat img_gray_left, img_gray_right;
  cv::cvtColor(dataset_[idx].img_[kCam_Left], img_gray_left, CV_BGR2GRAY);
  cv::cvtColor(dataset_[idx].img_[kCam_Right], img_gray_right, CV_BGR2GRAY);
  
  // Call both constructor.
  cv::Ptr<cv::StereoBM> sbm1 = cv::StereoBM::create(0, param.stereo_win_size_);
  cv::Ptr<cv::StereoSGBM> sbm2 = cv::StereoSGBM::create(0, param.stereo_num_disparities_,
                                                        param.stereo_block_size_,
                                                        param.stereo_smooth_p1_,
                                                        param.stereo_smooth_p2_);
  
  // Measure time - start.
  auto t_start = std::chrono::high_resolution_clock::now();
   
  // Caculate the disparity between two images.
  if(param.stereo_mode_ == kStereo_BM) {
    sbm1->compute(img_gray_left, img_gray_right, img_disparity16S);
  } else {
    sbm2->compute(img_gray_left, img_gray_right, img_disparity16S);
  }  
  
  // Measure time - end.
  auto t_end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t_end - t_start ).count();

  if(debug) {  
    std::cout << "Processing time: " << std::setprecision(4) << duration << "ms" << std::endl; 
  }
  
  // Image for debugging.
  normalize(img_disparity16S, img_disparity8, 0, 255, CV_MINMAX, CV_8U);
  
  // For each bbox, calculate distance (meter) and angle from the ego-vehicle, and update.
  for(auto i = 0; i < dataset_[idx].detection_.size(); ++i) {
    // Get bbox.
    cv::Rect box = dataset_[idx].detection_[i].GetBbox();
    
    // Get average intensity.
    const float avg_intensity = CalculateAvgIntensity(img_disparity16S, box);
    
    // Calculate distance from the ego-vehicle.
    // F-number is given in pixel unit.
    const float distance = param.eval_cam_baseline_ * param.eval_cam_fnum_ / (avg_intensity / disp_scale);
    
    // Get angle.
    const float angle = CalculateAngle(param.eval_cam_fov_, img_gray_left.cols, box);
    
    // Calculate coordinates.
    float x = distance * sin(DegToRad(angle));
    float z = distance * cos(DegToRad(angle));
    
    // Update.
    dataset_[idx].detection_[i].Set3DLoc(x, 0, z);
    
    // For debugging.
    if(debug && dataset_[idx].detection_[i].GetScore() 
       > param.save_image_disparity_score_min_) {
      // Draw bbox.
      cv::rectangle(img_disparity8, cv::Point(box.x, box.y), 
                    cv::Point(box.x + box.width, box.y + box.height),
                    cv::Scalar(255), 2, 8, 0);
      // Write intensity and angle.      
      std::stringstream stream;
      stream << "avg:" << std::fixed << std::setprecision(2) << avg_intensity
        << " deg:" << angle;
      std::string text = stream.str();
      
      WriteText(img_disparity8, box.x, box.y, text);
    }    
  }
  
  // Save images.
  if(debug) {
    // Get file name and remove extension.
    std::string file_name = save_dir + PathSeparator() + dataset_[idx].file_name_;
    file_name.erase(file_name.end()-4, file_name.end());
    
    std::string file_img;    
    file_img = file_name + "_grey_left.png";
    cv::imwrite(file_img.c_str(), img_gray_left);    
    file_img = file_name + "_grey_right.png";
    cv::imwrite(file_img.c_str(), img_gray_right);    
    file_img = file_name + "_disparity.png";
    cv::imwrite(file_img.c_str(), img_disparity16S);
    file_img = file_name + "_normalize.png";
    cv::imwrite(file_img.c_str(), img_disparity8);
  }

  return true;
}
  
// Calculate average of intensity values within the input box area.
float DataSet::CalculateAvgIntensity(cv::Mat img, cv::Rect box) {
  // Exceptions.
  if(box.width < 0 || box.height < 0) {
    ErrMsg(__func__, " Invalid box was detected.");
    return -1.;
  }
  
  const unsigned int left = cv::max(box.x, 0);
  const unsigned int right = cv::min(box.x + box.width, img.cols);
  const unsigned int top = cv::max(box.y, 0);
  const unsigned int bottom = cv::min(box.y + box.height, img.rows);
  
  double average = 0.;
  unsigned int count = 0;
  for(auto row = top; row < bottom; ++row) {
    for(auto col = left; col < right; ++col) {
      int val = img.at<short>(row, col);
      // Don't count if the value is zero.
      if(val <= 0) {
        continue;
      }
      average = (average * count + val) / (double)(count + 1);
      count++;
    }
  }
  
  return (float)average;  
}
  
// Calculate angle of the vehicle from the ego-vehicle.
float DataSet::CalculateAngle(const float h_fov_deg, const unsigned int img_width, 
                              cv::Rect box) {
  const float fov_half_rad = DegToRad(h_fov_deg / 2.);
  const float img_width_half = img_width / 2.;
  const float box_center = box.x + box.width / 2.;
  
  const float angle_rad = atan(tan(fov_half_rad)*(box_center - img_width_half)/img_width_half);
  
  return RadToDeg(angle_rad);
}
  
// Write GT-vehicles into the VOC output file.
void DataSet::SaveVOC(std::ofstream& out_file,
                      const std::vector<Vehicle>& vehicles,
                      unsigned int& seq,
                      std::string img_file_name, 
                      const int mode) {
  if(vehicles.size() < 1) {
    return;
  }
  
  // VOC format.
  //# [Number]
  //[Image file name]
  //[Number of objects]
  //[Type]  [left]  [top] [right] [bottom]  [difficulty]
  // Example of VOC format.
  //# 4931
  //009805.jpg
  //1  
  //8   97    133   392   276   0
  out_file << "# " << seq++ << std::endl;
  out_file << img_file_name << std::endl;
  out_file << vehicles.size() << std::endl;
  for(auto it = vehicles.begin(); it != vehicles.end(); ++it) {
    cv::Rect rect = (*it).GetBbox();
    out_file << VOC_LABEL_CAR 
        << "   " << rect.x << "\t" << rect.y << "\t";
    if(mode == kTopLeftBottomRight) {
      out_file << rect.width << "\t" << rect.height 
        << "\t" << 0 << std::endl;
    } else if (mode == kTopLeftWidthHeight) {
      out_file << rect.x + rect.width << "\t" << rect.y + rect.height 
        << "\t" << 0 << std::endl;
    } else {
      return;
    }
  }
}
  
  
} // namespace vehicle_detector
} // namespace robotics
