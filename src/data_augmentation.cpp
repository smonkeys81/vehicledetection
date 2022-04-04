// Copyright © 2019 Robotics, Inc. All Rights Reserved.

// This file provides functions to augment KITTI dataset.


#include "data_augmentation.h"
#include <opencv2/highgui/highgui.hpp>


namespace robotics {
namespace vehicle_detector {
  

// Constructor.
DataAugmentation::DataAugmentation(std::string config_file) : VehicleDetector(config_file) {
}
  

// Delete all previously augmented images and labels from the specified directories.
bool DataAugmentation::AugmentData() {
  // Load parameters.
  if(!LoadParamAugment(param_.file_data_augmentation_)) {
    ErrMsg(__func__, "Loading parameter failed:", param_.file_data_augmentation_);
    return false;
  }
  
  const std::string path_label = param_.dir_dataset_train_label_[kKITTI];
  const std::string path_image = param_.dir_dataset_train_image_[kKITTI];
    
  // Clear previously augmented data.
  if(!RemoveAugmentedFiles(path_label, path_image)) {
    ErrMsg(__func__, "Removing previous data failed.");
    return false;
  }
  
  // Load KITTI dataset.
  if(!kitti_.LoadDataSet(path_label, path_image)) {
    ErrMsg(__func__, "Loading KITTI dataset failed.");
    return false;
  }
  
  // Generate augmented data.
  if(!GenerateAugmentedFiles(path_label, path_image)) {
    ErrMsg(__func__, "Generating augmented data failed.");
    return false;
  }  
  
  return true;
}
 
// Delete all previously augmented images and labels from the specified directories.
bool DataAugmentation::RemoveAugmentedFiles(const std::string& path_label,
                                            const std::string& path_image) {
  // Check directories.
  DIR *dir_label, *dir_image;
  struct dirent *entry;
  // Check directories.
  if(!OpenDirectory(path_label, &dir_label, __func__)) {
    return false;
  }
  if(!OpenDirectory(path_image, &dir_image, __func__)) {
    closedir(dir_label);
    return false;
  }

  // Delete all augmented label files.
  while ((entry = readdir(dir_label)) != nullptr) {
    // Skip file "." and "..".
    if(!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, "..")) {
      continue;
    }
    
    // Find substring in file name and delete if the substring exists.
    std::string file = path_label + PathSeparator() + entry->d_name;
    if (file.find(aug) != std::string::npos) {
      // Delete file.
      std::remove(file.c_str());
    }
  }
  // Close dir.
  closedir(dir_label);
  
  // Delete all augmented image files.
  while ((entry = readdir(dir_image)) != nullptr) {
    // Skip file "." and "..".
    if(!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, "..")) {
      continue;
    }
    
    // Find substring in file name and delete if the substring exists.
    std::string file = path_image + PathSeparator() + entry->d_name;
    if (file.find(aug) != std::string::npos) {
      // Delete file.
      std::remove(file.c_str());
    }
  }
  // Close dir.
  closedir(dir_image);
  
  return true;
}
  
// Perform data augmentation on the specified directory.
bool DataAugmentation::GenerateAugmentedFiles(const std::string& path_label,
                                              const std::string& path_image) {
  if(kitti_.dataset_.size() < 1) {
    ErrMsg(__func__, "No available data exists.");
    return true;
  }
  
  // Augment for each image and for each augmentation method.
  for(int i = 0; i < kitti_.dataset_.size(); ++i) {
    for(int j = 0; j < kAugment_num; ++j) {
      // Check flag first, then randomly decide wheter perform augmentation or not.
      if(GetRandomZeroOne(aug_param_[j].prob_) && aug_param_[j].do_) {
        // Hard copy image.
        cv::Mat img_augmented = kitti_.dataset_[i].img_[kCam_Left].clone();
        std::vector<Vehicle> gt_bboxes;
        
        // Extract file name w/o extension.
        std::string file_name = kitti_.dataset_[i].file_name_;
        file_name.erase(file_name.end()-4, file_name.end());
        // Attach tag and properties to the file name.
        file_name += aug + std::to_string(j);
        
        // Augment image and adjust label.
        switch(j) {
          case kHorflip:
            // Horizontal flip will be implemented later, because this augmentation is provided in the Faster-RCNN of Caffe.
            break;
          case kTranslation:
            ImageTranslate(img_augmented, kitti_.dataset_[i], gt_bboxes, file_name);
            break;
          case kRotation:
            ImageRotate(img_augmented, kitti_.dataset_[i], gt_bboxes, file_name);
            break;
          case kScale:
            ImageScale(img_augmented, kitti_.dataset_[i], gt_bboxes, file_name);
            break;
          case kBrightness:
            ImageBrightness(img_augmented, kitti_.dataset_[i], gt_bboxes, file_name);
            break;
          case kBlur:
            ImageBlur(img_augmented, kitti_.dataset_[i], gt_bboxes, file_name);
            break;
          case kNoise:
            ImageNoise(img_augmented, kitti_.dataset_[i], gt_bboxes, file_name);
            break;
          case kCutout:
            ImageCutout(img_augmented, kitti_.dataset_[i], gt_bboxes, file_name);
            break;  
          default:
            ErrMsg(__func__, "Something went wrong.");
            break;
        }
          
        // Save image.
        std::string file_image = file_name + ".png";
        cv::imwrite(path_image + file_image, img_augmented);
        
        // Save label.
        std::string file_label = file_name + ".txt";
        std::ofstream out_file;
        out_file.open(path_label + file_label);
        if(!out_file) {
          ErrMsg(__func__, "Unable to open out file:", path_label + file_label);
          return false;
        }
        for(int k = 0; k < gt_bboxes.size(); ++k) {
          out_file << gt_bboxes[k].GetLabel() << " "; // Type.
          out_file << 0 << " "; // Non-truncated.
          out_file << 0 << " "; // Fully visible.
          out_file << 0 << " "; // Observation angle.
          cv::Rect_<float> box = gt_bboxes[k].GetBbox();
          out_file << box.x << " "; // Bbox - left.
          out_file << box.y << " "; // Bbox - top.
          out_file << box.x + box.width << " "; // Bbox - right.
          out_file << box.y + box.height << " "; // Bbox - bottom.
          out_file << 0.00 << " "; // 3D object dimension.
          out_file << 0.00 << " "; // 3D object dimension.
          out_file << 0.00 << " "; // 3D object dimension.
          out_file << 0.00 << " "; // 3D object location in camera coordinate.
          out_file << 0.00 << " "; // 3D object location in camera coordinate.
          out_file << 0.00 << " "; // 3D object location in camera coordinate.
          out_file << 0.00 << std::endl; // Confidence in detection.
        }
        out_file.close();
      }
    }
    if(i != 0 && i % 500 == 0) {
      std::cout << i << " images completed." << std::endl;
    }
  }
  
  return true;
}

// Image manipulation - translate.
bool DataAugmentation::ImageTranslate(cv::Mat& img, const DataImageLabel& label,
                                      std::vector<Vehicle>& gt_bboxes,
                                      std::string& file_name) {
  // Generate random number x and y for translation.
  // x and y are randomly chosen in {-max, -min, min, max}.
  const int zero_one_x1 = GetRandomZeroOne();
  const int zero_one_y1 = GetRandomZeroOne();
  // Min or max is chosen as the amount of translation. 
  float trans_x = (1 - zero_one_x1) * aug_param_[kTranslation].min_
    + zero_one_x1 * aug_param_[kTranslation].max_;
  float trans_y = (1 - zero_one_y1) * aug_param_[kTranslation].min_
    + zero_one_y1 * aug_param_[kTranslation].max_;
  // Use below to choose positive/negative
  trans_x = trans_x * (1 - 2 * GetRandomZeroOne());
  trans_y = trans_y * (1 - 2 * GetRandomZeroOne());
  
  // Translate image.
  cv::Mat trans_mat = (cv::Mat_<float>(2, 3) << 1, 0, trans_x, 0, 1, trans_y);
  cv::warpAffine(img, img, trans_mat, img.size());
  
  // Copy data.
  gt_bboxes = label.gt_vehicles_;
  copy(label.gt_dontcare_.begin(), label.gt_dontcare_.end(), 
       back_inserter(gt_bboxes));
  
  // Update labels.
  for(int i = 0; i < gt_bboxes.size(); ++i) {
    cv::Rect_<float> box = gt_bboxes[i].GetBbox();
    // Adjust width and height of bbox touching boundary.
    if(box.x + trans_x < 0) {
      box.width += (box.x + trans_x);
    }
    if(box.y + trans_y < 0) {
      box.height += (box.y + trans_y);
    }
    box.x = SetBoundary(box.x + trans_x, 0., img.cols);
    box.y = SetBoundary(box.y + trans_y, 0., img.rows);
    box.width = SetBoundary(box.x + box.width, 0., img.cols + trans_x) - box.x;
    box.height = SetBoundary(box.y + box.height, 0., img.rows + trans_y) - box.y;
    gt_bboxes[i].SetBbox(box);
  }
  
  // Update file name.
  file_name += "_x_" + std::to_string(trans_x) + "_y_" + std::to_string(trans_y);
  
  return true;
}

// Rotate bbox.
void DataAugmentation::BoxRotate(const cv::Point_<float> center,
                                 const float deg, cv::Rect_<float>& box) {
  // Get four corners - left-top, right-top, left-bottom, and right-bottom.
  cv::Point_<float> fbox[4];
  fbox[kROI_LeftTop].x = box.x;
  fbox[kROI_LeftTop].y = box.y;
  fbox[kROI_RightTop].x = box.x + box.width;
  fbox[kROI_RightTop].y = box.y;
  fbox[kROI_LeftBottom].x = box.x;
  fbox[kROI_LeftBottom].y = box.y + box.height;
  fbox[kROI_RightBottom].x = box.x + box.width;
  fbox[kROI_RightBottom].y = box.y + box.height;
  
  const float rad = DegToRad(deg);
  
  // Calculate rotation.
  for(int i = 0; i < kROI_NumCorners; ++i) {
    const float x = center.x 
      + (fbox[i].x - center.x) * cos(rad) + (fbox[i].y - center.y) * sin(rad);
    const float y = center.y 
      - (fbox[i].x - center.x) * sin(rad) + (fbox[i].y - center.y) * cos(rad);
    fbox[i].x = x;
    fbox[i].y = y;
  }
  
  // Since the box above may have slope, let's find new boundaries.
  float x_min = center.x * 2;
  float x_max = 0;
  float y_min = center.y * 2;
  float y_max = 0;
  for(int i = 0; i < kROI_NumCorners; ++i) {
    x_min = std::min(x_min, fbox[i].x);
    x_max = std::max(x_max, fbox[i].x);
    y_min = std::min(y_min, fbox[i].y);
    y_max = std::max(y_max, fbox[i].y);    
  }
  // Check image boundaries.
  x_min = SetBoundary(x_min, 0, center.x * 2);
  x_max = SetBoundary(x_max, 0, center.x * 2);
  y_min = SetBoundary(y_min, 0, center.y * 2);
  y_max = SetBoundary(y_max, 0, center.y * 2);
  
  // Assign, finally.
  box.x = x_min;
  box.y = y_min;
  box.width = x_max - x_min;
  box.height = y_max - y_min;
}
  
// Image manipulation - rotate.
bool DataAugmentation::ImageRotate(cv::Mat& img, const DataImageLabel& label,
                                   std::vector<Vehicle>& gt_bboxes,
                                   std::string& file_name) {
  // Randomly choose min or max to rotate. 
  const int zero_one = GetRandomZeroOne();
  float deg = (1 - zero_one) * aug_param_[kRotation].min_
    + zero_one * aug_param_[kRotation].max_;
  deg = deg * (1 - 2 * GetRandomZeroOne());
  
  // Rotate image.
  cv::Point2f pc(img.cols/2., img.rows/2.);
  cv::Mat rot = cv::getRotationMatrix2D(pc, deg, 1.0);
  cv::warpAffine(img, img, rot, img.size());
  
  // Copy data.
  gt_bboxes = label.gt_vehicles_;
  copy(label.gt_dontcare_.begin(), label.gt_dontcare_.end(), 
       back_inserter(gt_bboxes));
  
  // Update labels.
  for(int i = 0; i < gt_bboxes.size(); ++i) {
    cv::Rect_<float> box = gt_bboxes[i].GetBbox();
    BoxRotate(cv::Point(img.cols/2., img.rows/2.), deg, box);
    gt_bboxes[i].SetBbox(box);
  }
  
  // Update file name.
  file_name += "_deg_" + std::to_string(deg);
  
  return true;
}

// Image manipulation - scale.
bool DataAugmentation::ImageScale(cv::Mat& img, DataImageLabel& label,
                                   std::vector<Vehicle>& gt_bboxes,
                                   std::string& file_name) {
  // Randomly choose min or max to magnify. 
  const int zero_one = GetRandomZeroOne();
  float mag = (1 - zero_one) * aug_param_[kScale].min_
    + zero_one * aug_param_[kScale].max_;
  
  // Adjust scale of the input image.
  cv::Mat img_resized;
  cv::resize(img, img_resized, cv::Size(), mag, mag);
  const float margin_w = img.cols * fabs(mag - 1.0) / 2.;
  const float margin_h = img.rows * fabs(mag - 1.0) / 2.;
  float border_left = 0.;
  float border_right = img.cols;
  float border_top = 0.;
  float border_bottom = img.rows;
  if(mag >= 1.0) {
    // Copy ROI.
    cv::Rect rect(margin_w, margin_h, img.cols, img.rows);
    img = img_resized(rect);
  } else {
    cv::copyMakeBorder(img_resized, img, margin_h, margin_h,
                       margin_w, margin_w, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
    border_left = margin_w;
    border_right = img.cols - margin_w;
    border_top = margin_h;
    border_bottom = img.rows - margin_h;
  }
  
  // Copy data.
  gt_bboxes = label.gt_vehicles_;
  copy(label.gt_dontcare_.begin(), label.gt_dontcare_.end(), 
       back_inserter(gt_bboxes));
  
  // Update labels.
  for(int i = 0; i < gt_bboxes.size(); ++i) {
    cv::Rect_<float> box = gt_bboxes[i].GetBbox();
    box.x = img.cols/2. + (box.x - img.cols/2.) * mag;
    box.y = img.rows/2. + (box.y - img.rows/2.) * mag;
    box.width *= mag;
    box.height *= mag; 
    // Adjust width and height of bbox touching boundary.
    if(box.x < 0) {
      box.width += box.x;
    }
    if(box.y < 0) {
      box.height += box.y;
    }
    // Boundary check.
    box.x = SetBoundary(box.x, border_left, border_right);
    box.y = SetBoundary(box.y, border_top, border_bottom);
    box.width = SetBoundary(box.x + box.width, border_left, border_right) - box.x;
    box.height = SetBoundary(box.y + box.height, border_top, border_bottom) - box.y;
    gt_bboxes[i].SetBbox(box);
  }
  
  // Update file name.
  file_name += "_scale_" + std::to_string(mag);
  
  return true;
}
  
// Image manipulation - brightness.
bool DataAugmentation::ImageBrightness(cv::Mat& img, DataImageLabel& label,
                                       std::vector<Vehicle>& gt_bboxes,
                                       std::string& file_name) {
  // Randomly choose min or max.
  const int zero_one = GetRandomZeroOne();
  float intensity = (1 - zero_one) * aug_param_[kBrightness].min_
    + zero_one * aug_param_[kBrightness].max_;
  // Randomly choose + or -.
  intensity *= (1 - 2 * GetRandomZeroOne());
  
  // Change brightness.
  img.convertTo(img, -1, 1, intensity);
  
  // Copy data.
  gt_bboxes = label.gt_vehicles_;
  copy(label.gt_dontcare_.begin(), label.gt_dontcare_.end(), 
       back_inserter(gt_bboxes));
  
  // No need to transform bboxes.
  
  // Update file name.
  file_name += "_br_" + std::to_string(intensity);
  
  return true;
}

// Image manipulation - blurring.
bool DataAugmentation::ImageBlur(cv::Mat& img, DataImageLabel& label,
                                 std::vector<Vehicle>& gt_bboxes,
                                 std::string& file_name) {
  // Choose sigma.
  const int zero_one = GetRandomZeroOne();
  float sigma = (1 - zero_one) * aug_param_[kBlur].min_
    + zero_one * aug_param_[kBlur].max_;
  
  // Blurring operation.
  cv::GaussianBlur(img, img, cv::Size(0, 0), sigma, sigma);
  
  // Copy data.
  gt_bboxes = label.gt_vehicles_;
  copy(label.gt_dontcare_.begin(), label.gt_dontcare_.end(), 
       back_inserter(gt_bboxes));
  
  // No need to transform bboxes.
  
  // Update file name.
  file_name += "_blur_sigma" + std::to_string(sigma);
  
  return true;
}
  
// Image manipulation - noise.
bool DataAugmentation::ImageNoise(cv::Mat& img, DataImageLabel& label,
                                  std::vector<Vehicle>& gt_bboxes,
                                  std::string& file_name) {
  // Randomly choose min or max.
  const int zero_one = GetRandomZeroOne();
  float var = (1 - zero_one) * aug_param_[kNoise].min_
    + zero_one * aug_param_[kNoise].max_;
  
  // Add noise.
  cv::Mat noise(img.size(), img.type());
  cv::randn(noise, 0, var);
  img += noise;
  
  // Copy data.
  gt_bboxes = label.gt_vehicles_;
  copy(label.gt_dontcare_.begin(), label.gt_dontcare_.end(), 
       back_inserter(gt_bboxes));
  
  // No need to transform bboxes.
  
  // Update file name.
  file_name += "_noise_var" + std::to_string(var);
  
  return true;
}

// Image manipulation - cutout.
bool DataAugmentation::ImageCutout(cv::Mat& img, DataImageLabel& label,
                                   std::vector<Vehicle>& gt_bboxes,
                                   std::string& file_name) {
  // We don't use max and randomly here.
  // Instead, we use the bigger one for big bbox.
  // The height of the big box should be greater than (large-cutting-size) * (custom ratio)
  const float cut_size_large = aug_param_[kCutout].max_;
  const float cut_size_small = aug_param_[kCutout].min_;
  const float ratio = aug_param_[kCutout].custom_;
  
  // Copy data.
  gt_bboxes = label.gt_vehicles_;
  copy(label.gt_dontcare_.begin(), label.gt_dontcare_.end(), 
       back_inserter(gt_bboxes));
  
  // Sort bboxes by x coordinate.
  sort(gt_bboxes.begin(), gt_bboxes.end(),
       [](Vehicle const& v1, Vehicle const& v2){return v1.GetLeft() < v2.GetLeft();} );
  
  // Recent study has shown that image and object-aware random erasing (cutting out)
  // resulted in better performance than image-aware or object-aware random erasing.
  // (Please see https://arxiv.org/pdf/1708.04896.pdf.)
  
  // First, for each bbox, erase part of it. (Object-aware)
  // Choose the position in the bbox randomly.
  // But skip erasing if the bbox is too small to erase.
  for(int i = 0; i < gt_bboxes.size(); ++i) {
    unsigned int cut_size;
    cv::Rect_<float> box = gt_bboxes[i].GetBbox();
    if(box.height > cut_size_large * ratio) {
      cut_size = cut_size_large;                  
    } else if (box.height > cut_size_small * ratio) {
      cut_size = cut_size_small;  
    } else {
      // Skip otherwise.
      continue;
    }
    
    // Determine the location in the box randomly.
    const int center_x = GetRandomReal(box.x, box.x + box.width);
    const int center_y = GetRandomReal(box.y, box.y + box.height);
    // Make RoI rectangle.
    cv::Rect rect(center_x-cut_size/2., center_y-cut_size/2., cut_size, cut_size);
    cv:rectangle(img, rect, cv::Scalar::all(0), CV_FILLED);
  }
  // Second, find adjacent two bboxes and erase the area that contains some of the
  // area of the bboxes.
  // To make this simple, let's skip erasing if two adjacent boxes are overlapped.
  // This probably prevent erasing too much area in the bbox. 
  for(int i = 0; i < gt_bboxes.size(); ++i) {
    // Check if there's next element.
    if(i + 1 >= gt_bboxes.size()) {
      continue;
    }
    
    // Skip if two boxes are overlapped.
    cv::Rect_<float> box_1 = gt_bboxes[i].GetBbox();
    cv::Rect_<float> box_2 = gt_bboxes[i+1].GetBbox();
    if(0 != CalculateIoU(box_1, box_2)) {
      continue;
    }
      
    cv::Rect rect;
    const unsigned int thickness = 10;
    // When two boxes are located side to side.
    if(box_1.y < box_2.y && box_1.y + box_1.height > box_2.y) {
      const unsigned int y = (box_2.y + box_1.y + box_1.height) / 2;
      const unsigned int left = box_1.x + box_1.width - 5;
      const unsigned int cut_size = (box_2.x + 5) - left;
      rect = cv::Rect(left, y - thickness/2, cut_size, thickness);
    } else if (box_1.y < box_2.y + box_2.height 
               && box_1.y + box_1.height > box_2.y + box_2.height) {
      const unsigned int y = (box_1.y + box_2.y + box_2.height) / 2;
      const unsigned int left = box_1.x + box_1.width - 5;
      const unsigned int cut_size = (box_2.x + 5) - left;
      rect = cv::Rect(left, y - thickness/2, cut_size, thickness);
    // When two boxes are located up and down.
    } else if (box_1.x < box_2.x && box_1.x + box_1.width > box_2.x) {
      const unsigned int x = (box_1.x + box_2.x + box_1.width) / 2;
      // Box 2 is above box 1.
      if(box_1.y > box_2.y + box_2.height) {        
        const unsigned int top = box_2.y + box_2.height - 5;
        const unsigned int cut_size = (box_1.y + 5) - top;   
        rect = cv::Rect(x - thickness/2, top, thickness, cut_size);  
      // Box 1 is above box 2.
      } else {
        const unsigned int top = box_1.y + box_1.height - 5;
        const unsigned int cut_size = (box_2.y + 5) - top;
        rect = cv::Rect(x - thickness/2, top, thickness, cut_size);
      }            
    } else {
      continue;
    }
    // Draw Rectangle.
    cv::rectangle(img, rect, cv::Scalar::all(0), CV_FILLED); 
    
    // Increase iterator one more.
    ++i;
  }
  
  
  
  // No need to transform bboxes.
  
  // Update file name.
  file_name += "_cut";
  
  return true;
}
  
  
} // namespace data_augmentor
} // namespace robotics
