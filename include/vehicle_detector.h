// Copyright © 2019 Robotics, Inc. All Rights Reserved.

// This file provides data structure and methods for the vehicle detection pipeline.


#ifndef ROBOTICS_VEHICLEDETECTION_VEHICLEDETECTOR_H_
#define ROBOTICS_VEHICLEDETECTION_VEHICLEDETECTOR_H_


#include "util.h"
#include "param/param.h"
#include "vehicle.h"
#include <caffe/caffe.hpp>
#include <caffe/FRCNN/util/frcnn_param.hpp>
#include <caffe/FRCNN/util/frcnn_helper.hpp>
#include <vector>


namespace robotics {
namespace vehicle_detector {


// Define bbox colors for ground-truth, prediction and don't care.
// Other functions may use these colors too.
const cv::Scalar color_red = CV_RGB(255, 0, 0);
const cv::Scalar color_blue = CV_RGB(0, 0, 255);
const cv::Scalar color_gray = CV_RGB(128, 128, 128);
  
// Define channels and the maximum number of channels.
enum ENUM_CHANNEL {
  kCh_0 = 0,
  kCh_1 = 1,
  kCh_2 = 2,
  kNum_Ch_Grey = 1,
  kNum_Ch_RGB = 3,
};
  
// Define how to normalize input image.
enum ENUM_IMAGE_NORMALIZE {
  kImg_Norm_Subtract_Mean = 0,
  kImg_Norm_Subtract_Divide_128,
  kImg_Norm_Divide_256,
};

// Meaning of index for img_info resized.
enum ENUM_IMG_INFO {
  kImg_Resized_Height = 0,
  kImg_Resized_Width,
  kImg_Resized_Scale,
  kImg_Resized_Info_Size,
};

// Define size of data structure from the network.
enum ENUM_FRCNN_DATA_SIZE {
  kNum_Data_Bbox = 4,
  kNum_Data_Pred = 5,  // same as caffe::Frcnn::DataPrepare::NUM
};

// Define index in roi data, including bbox and score.
enum ENUM_ROI_DATA {
  kROI_Left = 0,
  kROI_Top,
  kROI_Right,
  kROI_Bottom,
  kROI_Score,
  kROI_Width = kROI_Right,
  kROI_Height = kROI_Bottom,
};

// Define indices of four bbox corners.
enum ENUM_ROI_FOUR_CORNERS {
  kROI_LeftTop = 0, 
  kROI_RightTop,
  kROI_LeftBottom,
  kROI_RightBottom,
  kROI_NumCorners,
};
  
// Label number for vehicles.
enum ENUM_LABEL_VEHICLE {
  kLabel_Bus = 6,
  kLabel_Car = 7,
};
  
// Bbox types.
enum ENUM_BBOX_TYPE {
  kBbox_GT = 0,
  kBbox_pred,
  kBbox_dontcare,
};
  
// Bbox infos - this is to choose which infomation to write on the top of bbox.
enum ENUM_BBOX_INFO {
  kBbox_Score = 0,
  kBbox_Pos,
};
  
// Bbox types.
enum ENUM_SHAPE_INDEX {
  kShape_Kernel = 1,
  kShape_Height,
  kShape_Width,
};

// Forward declaration for unit test.
class VehicleDetectorTest;

/// \class VehicleDetector
/// This is a class to provide vehicle detection methods.
class VehicleDetector {
friend class VehicleDetectorTest;
public:
  /// \brief Constructor.
  /// \param[in] config_file Path to configuration file.
  VehicleDetector(const std::string& config_file);

  /// \brief Destructor.
  ~VehicleDetector() {}

  /// \brief Method to train a model.
  /// \param[in] model_name Name of the backbone model.
  /// \param[in] snapshot_file Snapshot file to resume training from the previous training.
  /// \return True if all training procedure is completed successfully, false otherwise.
  bool TrainModelByName(const std::string& model_name, const std::string& snapshot_file = "");
  
  /// \brief Method to train a model.
  /// \param[in] solver_file Solver file including the path to train model and hyper-parameters.
  /// \param[in] pretrained_file Pretrained weights.
  /// \param[in] snapshot_file Snapshot file to resume training from the previous training.
  /// \return True if all training procedure is completed successfully, false otherwise.
  bool TrainModel(const std::string& solver_file, const std::string& pretrained_file,
                  const std::string& snapshot_file ="");
  
  /// \brief Method loading a model.
  /// \param[in] model_name Name of the backbone model.
  /// \param[in] trained_file Trained file including parameter values.
  /// \return True if all files were loaded successfully, false otherwise.
  bool LoadModelByName(const std::string& model_name, const std::string& trained_file);
  
  /// \brief Method loading a model.
  /// \param[in] model_file Model file.
  /// \param[in] trained_file Trained file including parameter values.
  /// \return True if all files were loaded successfully, false otherwise.
  bool LoadModelByFile(const std::string& model_file, const std::string& trained_file);

  /// \brief Method to set mean value of images by mean value of each channel, e.g., googlenet doesn't use mean file.
  /// \param[in] num_ch Number of channels.
  /// \param[in] ch0 Mean value of channel 0.
  /// \param[in] ch1 Mean value of channel 1.
  /// \param[in] ch2 Mean value of channel 2.
  /// \return True if mean file is correctly loaded, false otherwise.
  bool SetMean(const unsigned short num_ch, const float ch0, const float ch1 = 0, const float ch2 = 0);

  /// \brief Perform detection process from the input image.
  /// \param[in] img Input image.
  /// \param[in] debug_msg Print debug message.
  /// \param[out] detected_vehicles Results of the detection. 
  /// \return True if the function worked well, false when exception occurs.
  bool Detect(const cv::Mat img, const bool debug_msg,
              std::vector<Vehicle>& detected_vehicles);

  /// \brief Visualize detection data on the image.
  /// \param[in] img Input image.
  /// \param[in] vehicles Information to be displayed.
  /// \param[in] type Type of bbox.
  /// \param[in] info Information type to be displayed - score or coordinates.
  void VisualizeDetections(cv::Mat& img, std::vector<Vehicle> vehicles,
                          ENUM_BBOX_TYPE type, ENUM_BBOX_INFO info);
  
  /// \brief Set number of input to Faster RCNN network.
  /// \param[in] num_input Number of input.
  void SetNumInputFrcnn(const int num_input) { num_input_faster_rcnn_ = num_input; }
  
  /// \brief Set number of output to Faster RCNN network.
  /// \param[in] num_output Number of output.
  void SetNumOutputFrcnn(const int num_output) { num_output_faster_rcnn_ = num_output; }
  
  /// \brief Get number of input to Faster RCNN network.
  /// \return Number of intput.
  int GetNumInputFrcnn() { return num_input_faster_rcnn_; }
  
  /// \brief Get number of output from Faster RCNN network.
  /// \return Number of output.
  int GetNumOutputFrcnn() { return num_output_faster_rcnn_; }
  
  /// \brief Extract features from network and save.
  /// \param[in] layer_name Name of layer.
  /// \param[out] img_feat Container of feature images.
  void ConvFeatures(const std::string& layer_name, std::vector<cv::Mat>& img_feat);
  
  /// \brief Save all convolution images coming out from convolution layers.
  /// \param[in] path Path to save images.
  void SaveAllConvolutionImages(const std::string& path);
  
  /// \brief Calculate intersection over union (IoU).
  /// \param[in] bbox_pred Predicted bouding box.
  /// \param[in] bbox Ground-truth or don't-care bounding box.
  /// \param[in] box_type Type of bbox - Ground-truth or don't-care.
  /// \return IoU value between two bounding boxes.
  float CalculateIoU(cv::Rect_<float> bbox_pred, cv::Rect_<float> bbox,
                     ENUM_BBOX_TYPE box_type = kBbox_GT);
  
private:	
  /// \brief Method to preprocess the input image.
  /// \param[in] img Input image.
  /// \param[in] mode Method for input image normalization.
  /// \param[out] img_info Height, width, and scale of the image.
  /// \return True if preprocess is done successfully, false otherwise.
  /// By calling this function the input image is normalized by mean values, and resized if required.
  /// Preprocessed image is copied to data buffer which is input of the network.
  bool Preprocess(const cv::Mat img, ENUM_IMAGE_NORMALIZE mode,
                  float *img_info);

  /// \brief Sort RoIs by score, descending order.
  /// \param[in] num_rois Number of RoIs.
  /// \param[in] pred Predicted data consists of bboxes and scores.
  /// \param[out] sorted_pred Sorted prediction data by score.
  void SortBbox(const unsigned int num_rois, const float* pred, float* sorted_pred);
  
  /// \brief Method for inverse transform of bbox.
  /// \param[in] num_rois Number of RoIs.
  /// \param[in] num_classes Number of labels.py
  /// \param[in] box_deltas Delta value of the box in feature level.
  /// \param[in] pred_cls Predicted classes.
  /// \param[in] boxes Bboxes.
  /// \param[out] pred Predicted data consists of bboxes and labels.
  /// \param[in] img_height Height of the input image.
  /// \param[in] img_width Width of the input image.
  void InvTransformBbox(const unsigned int num_rois, const unsigned int num_classes,
      const float* box_deltas, const float* pred_cls, float* boxes, float* pred,
      const unsigned int img_height, const unsigned int img_width);

public:
  /// \brief Parameters for vehicle detection pipeline.
  Param param_;
  
private:
  /// \brief Caffe-based network data.
  std::shared_ptr<caffe::Net<float> > net_;

  /// \brief Width and height of input layer. 
  cv::Size input_geometry_;

  /// \brief Number of input channels.
  int num_channels_;

  /// \brief List of labels.
  std::vector<std::string> labels_;

  /// \brief Mean values of the image.
  float mean_[kNum_Ch_RGB];
  
  /// \brief Number of input to the network.
  int num_input_faster_rcnn_;
  
  /// \brief Number of output from the network.
  int num_output_faster_rcnn_;
};


} // namespace vehicle_detector
} // namespace robotics


#endif // ROBOTICS_VEHICLEDETECTION_VEHICLEDETECTOR_H_
