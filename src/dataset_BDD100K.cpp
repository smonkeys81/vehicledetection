// Copyright © 2020 Robotics, Inc. All Rights Reserved.

// This file provides definitions regarding BDD100K dataset.

#include "dataset_BDD100K.h"
#include <opencv2/highgui/highgui.hpp>


namespace robotics {
namespace vehicle_detector {


  // Load all images from the dataset.
bool DataSetBDD100K::LoadDataSet(const std::string& path_label,
                                 const std::string& path_img) {
  // Check directories.
  DIR *dir_img;
  struct dirent *entry;
  if(!OpenDirectory(path_img, &dir_img, __func__)) {
    return false;
  }
  // Close left image dir immediately.
  closedir(dir_img);
  
  // Load labels by opening Json file - error/exception message is printed in the function.
  Json::Value root;  
  if(!OpenFileJSON(path_label, root)) {
    return false;
  }
  
  for (int i = 0; i < root.size(); ++i) {
    const Json::Value label = root[i]["labels"];
    
    DataImageLabel single_data;
    
    for (int j = 0; j < label.size(); ++j) {
      const std::string category = root[i]["labels"][j]["category"].asString();
      
      if(!category.compare("car")) {
        Vehicle vehicle;
        
        // Get label.
        vehicle.SetLabel("Car");
        
        // Get bbox.
        float left = root[i]["labels"][j]["box2d"]["x1"].asFloat();
        float top = root[i]["labels"][j]["box2d"]["y1"].asFloat();
        float width = root[i]["labels"][j]["box2d"]["x2"].asFloat() - left;
        float height = root[i]["labels"][j]["box2d"]["y2"].asFloat() - top;
        cv::Rect_<float> rect(left, top, width, height);
        vehicle.SetBbox(rect);

        // Skip - Get 3D location.
        
        // Store vehicle data.
        single_data.gt_vehicles_.push_back(vehicle);
      }
    }
    
    // Load image.
    const std::string file_name = root[i]["name"].asString();
    single_data.file_name_ = file_name;
    const std::string img_path = path_img + file_name;
    single_data.img_[kCam_Left] = cv::imread(img_path, -1);
    dataset_.push_back(single_data);  
  }

  return true;
}
                               
// Convert the BDD100K data into VOC format.
bool DataSetBDD100K::ConvertVOC(const std::string& in_file,
                                const std::string& out_file) {
  // Open output file.
  std::ofstream f_out;
  f_out.open(out_file);
  if(!f_out) {
    ErrMsg(__func__, "Unable to open input file:", out_file);
    return false;
  }
  
  // Open Json file - error/exception message is printed in the function.
  Json::Value root;  
  if(!OpenFileJSON(in_file, root)) {
    return false;
  }
  
  // Example of BDD100K label in json file.
  //
  //[
  //  {
  //      "name": "b1c66a42-6f7d68ca.jpg",
  //      "attributes": {
  //          "weather": "overcast",
  //          "scene": "city street",
  //          "timeofday": "daytime"
  //      },
  //      "timestamp": 10000,
  //      "labels": [
  //          {
  //              "category": "car",
  //              "attributes": {
  //                  "occluded": false,
  //                  "truncated": false,
  //                  "trafficLightColor": "none"
  //              },
  //              "manualShape": true,
  //              "manualAttributes": true,
  //              "box2d": {
  //                  "x1": 819.464053,
  //                  "y1": 280.082505,
  //                  "x2": 889.23726,
  //                  "y2": 312.742305
  //              },
  //              "id": 51
  //          },
          
  unsigned int seq = 0;
  // Read, convert, and write.
  for (int i = 0; i < root.size(); ++i) {
    const std::string file_name = root[i]["name"].asString();
    const Json::Value label = root[i]["labels"];
    
    std::vector<Vehicle> vehicles;
    
    for (int j = 0; j < label.size(); ++j) {
      const std::string category = root[i]["labels"][j]["category"].asString();
      if(!category.compare("car")) {
        float left = root[i]["labels"][j]["box2d"]["x1"].asFloat();
        float top = root[i]["labels"][j]["box2d"]["y1"].asFloat();
        float right = root[i]["labels"][j]["box2d"]["x2"].asFloat();
        float bottom = root[i]["labels"][j]["box2d"]["y2"].asFloat();
        cv::Rect rect(left, top, right, bottom);
        
        // Store bbox.
        Vehicle v;
        v.SetBbox(rect);
        vehicles.push_back(v); 
      }
    }
    
    // Write labels in the output file.
    SaveVOC(f_out, vehicles, seq, file_name, kTopLeftBottomRight);
  }
  
  // Close file.
  f_out.close();
  
  return true;  
}
  
  
} // namespace vehicle_detector
} // namespace robotics
