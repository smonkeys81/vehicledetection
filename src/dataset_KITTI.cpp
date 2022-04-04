// Copyright © 2019 Robotics, Inc. All Rights Reserved.

// This file provides definitions regarding KITTI dataset.

#include "dataset_KITTI.h"
#include <opencv2/highgui/highgui.hpp>


namespace robotics {
namespace vehicle_detector {


// Load all images from the dataset.
bool DataSetKITTI::LoadDataSet(const std::string& path_label,
                               const std::string& path_img_left,
                               const std::string& path_img_right) {
  // Check directories.
  DIR *dir_label, *dir_img_left, *dir_img_right;
  struct dirent *entry;
  // Check directories.
  if(!OpenDirectory(path_label, &dir_label, __func__)) {
    return false;
  }
  if(!OpenDirectory(path_img_left, &dir_img_left, __func__)) {
    closedir(dir_label);
    return false;
  }
  // Close left image dir immediately.
  closedir(dir_img_left);
  if(path_img_right != "") {
    if(!OpenDirectory(path_img_right, &dir_img_right, __func__)) {
      closedir(dir_label);
      return false;
    }
    std::cout << "Right images directory was confirmed." << std::endl;
    // Close right image dir immediately.
    closedir(dir_img_right);
    // Update flag that right images exist.
    exist_img_right_ = true;    
  }
    
  // Load labels and images below.
  while ((entry = readdir(dir_label)) != nullptr) {
    // Skip file "." and "..".
    if(!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, "..")) {
      continue;
    }
    // Open label file.
    std::string full_path = path_label + PathSeparator() + entry->d_name;
    // Open and read each label file first.
    std::ifstream inFile;
    inFile.open(full_path);
    if(!inFile) {
      ErrMsg(__func__, "Unable to open label file:", full_path);
      closedir(dir_label);
      return false;
    }
    
    DataImageLabel single_data;
    
    // Read label file, line by line.
    // KITTI label example:
    // Car 0.00 1 3.03 486.51 174.21 531.22 191.03 1.59 1.76 4.18 -9.99 1.75 71.11 2.89.
    std::string str[kKITTI_num_data];
    while(inFile >> str[kKITTI_type] >> str[kKITTI_truncated] 
          >> str[kKITTI_occluded] >> str[kKITTI_alpha]
          >> str[kKITTI_bbox_left] >> str[kKITTI_bbox_top]
          >> str[kKITTI_bbox_right] >> str[kKITTI_bbox_bottom]
          >> str[kKITTI_3d_height] >> str[kKITTI_3d_width]
          >> str[kKITTI_3d_length] >> str[kKITTI_3d_loc_x]
          >> str[kKITTI_3d_loc_y] >> str[kKITTI_3d_loc_z]
          >> str[kKITTI_score]) {
      // Check type of object.
      if(!str[kKITTI_type].compare(KITTI_VEHICLE_CAR) 
         || !str[kKITTI_type].compare(KITTI_VEHICLE_VAN) 
         || !str[kKITTI_type].compare(KITTI_VEHICLE_TRUCK)
         || !str[kKITTI_type].compare(KITTI_DONTCARE)) {     
        
        Vehicle vehicle;
        
        // Get label.
        vehicle.SetLabel(str[kKITTI_type]);
        
        // Get bbox.
        float left = std::stof(str[kKITTI_bbox_left]);
        float top = std::stof(str[kKITTI_bbox_top]);
        float width = std::stof(str[kKITTI_bbox_right]) - left;
        float height = std::stof(str[kKITTI_bbox_bottom]) - top;
        cv::Rect_<float> rect(left, top, width, height);
        vehicle.SetBbox(rect);
        
        // Get 3D location.
        float x = std::stof(str[kKITTI_3d_loc_x]);
        float y = std::stof(str[kKITTI_3d_loc_y]);
        float z = std::stof(str[kKITTI_3d_loc_z]);
        vehicle.Set3DLoc(x, y, z);
        
        // Get difficulty - occlusion.
        vehicle.SetDifficulty(std::stoi(str[kKITTI_occluded]));
        
        // Store vehicle data.        
        if (!str[kKITTI_type].compare(KITTI_DONTCARE)
           || !str[kKITTI_type].compare(KITTI_MISC)) {
          single_data.gt_dontcare_.push_back(vehicle);
        } else {
          single_data.gt_vehicles_.push_back(vehicle);
        }
      }
    }
    inFile.close();
    
    // Load image.
    // Extract file name without extension and combine directory together.
    std::string file_name = entry->d_name;
    file_name.replace(file_name.size()-3, 3, "png");
    single_data.file_name_ = file_name;
    std::string img_path_left = path_img_left + file_name;
    single_data.img_[kCam_Left] = cv::imread(img_path_left, -1);  
    if(exist_img_right_ == true) {
      std::string img_path_right = path_img_right + file_name;
      single_data.img_[kCam_Right] = cv::imread(img_path_right, -1);  
    }
    dataset_.push_back(single_data);    
  }  
  // Close label dir.
  closedir(dir_label);
  
  return true;
}
  
// Convert the KITTI data into VOC format.
bool DataSetKITTI::ConvertVOC(const std::string& path_label,
                              const std::string& file_label) {
  // Access directories.
  DIR *dir_label;
  struct dirent *entry;
  // Check label directory.
  if(!OpenDirectory(path_label, &dir_label, __func__)) {
    return 0;
  }
  
  // Open output file.
  std::ofstream outFile;
  outFile.open(file_label);
  if(!outFile) {
    ErrMsg(__func__, "Unable to open out file:", file_label);
    closedir(dir_label);
    return 0;
  }
  
  unsigned int seq = 0;
  // Read, convert, and write.
  std::ifstream inFile;
  while ((entry = readdir(dir_label)) != nullptr) {
    // Skip "." and "..".
    if(!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, "..")) {
      continue;
    }
    // Open label file.
    std::string full_path = path_label + entry->d_name;
    inFile.open(full_path);
    if(!inFile) {
      ErrMsg(__func__, "Unable to open label file:", full_path);
      closedir(dir_label);
      return 0;
    }
    // Make image file name.
    std::string img_file_name = entry->d_name;
    img_file_name.replace(img_file_name.size()-3, 3, "png");
    
    std::vector<Vehicle> vehicles;
    std::string str[kKITTI_num_data];
    int num_data = kKITTI_type;
    // Read label file, line by line.
    // KITTI label example:
    // Car 0.00 1 3.03 486.51 174.21 531.22 191.03 1.59 1.76 4.18 -9.99 1.75 71.11 2.89.
    while(inFile >> str[kKITTI_type] >> str[kKITTI_truncated] 
          >> str[kKITTI_occluded] >> str[kKITTI_alpha]
          >> str[kKITTI_bbox_left] >> str[kKITTI_bbox_top]
          >> str[kKITTI_bbox_right] >> str[kKITTI_bbox_bottom]
          >> str[kKITTI_3d_height] >> str[kKITTI_3d_width]
          >> str[kKITTI_3d_length] >> str[kKITTI_3d_loc_x]
          >> str[kKITTI_3d_loc_y] >> str[kKITTI_3d_loc_z]
          >> str[kKITTI_score]) {
      // Check type of object.
      if(!str[kKITTI_type].compare(KITTI_VEHICLE_CAR) 
         || !str[kKITTI_type].compare(KITTI_VEHICLE_VAN) 
         || !str[kKITTI_type].compare(KITTI_VEHICLE_TRUCK)) { 
        // Get bbox.
        float left = std::stof(str[kKITTI_bbox_left]);
        float top = std::stof(str[kKITTI_bbox_top]);
        float right = std::stof(str[kKITTI_bbox_right]);
        float bottom = std::stof(str[kKITTI_bbox_bottom]);
        cv::Rect rect(left, top, right, bottom);
        
        // Store bbox.
        Vehicle v;
        v.SetBbox(rect);
        vehicles.push_back(v); 
      }
    }
    inFile.close(); 
    
    // Write labels in the output file.
    SaveVOC(outFile, vehicles, seq, img_file_name, kTopLeftBottomRight);
  }
  
  // Close file and directory.
  outFile.close();
  closedir(dir_label);
  
  return true;  
}
  
  
} // namespace vehicle_detector
} // namespace robotics
