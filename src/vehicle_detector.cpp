// Copyright © 2019 Robotics, Inc. All Rights Reserved.

// This file implements the methods for alpha version of vehicle detection pipeline.

#include "vehicle_detector.h"
#include <opencv2/highgui/highgui.hpp>


namespace robotics {
namespace vehicle_detector {


// Constructor.
VehicleDetector::VehicleDetector(const std::string& config_file) {
  // Load parameter.
  param_.LoadParam(config_file);

  // set GPU to use.
  if(!caffe::Caffe::CheckDevice(param_.device_gpu_num_)) {
    std::cerr << "The specified GPU does not exist!" << std::endl;
    throw;
  }
  
  // Initialize mean values to zero.
  for(int i = 0; i < kNum_Ch_RGB; ++i) {
    mean_[i] = 0.;
  }
}

// Method to train a model.
bool VehicleDetector::TrainModelByName(const std::string& model_name,
                                       const std::string& snapshot_file) {
  // Get index of backbone.
  unsigned int index_model;
  if(!param_.GetBackboneNum(model_name, index_model)) {
    ErrMsg(__func__, "Invalid backbone name", model_name);
    return false;
  }
  
  return TrainModel(param_.file_solver_[index_model],
                    param_.file_pretrain_[index_model],
                    snapshot_file);
}
  
// Method to train a model - overloaded function.
bool VehicleDetector::TrainModel(const std::string& solver_file,
                                 const std::string& pretrained_file,
                                 const std::string& snapshot_file) {
  // File check - solver.
  if(!FileExist(solver_file)) {
    ErrMsg(__func__, "Unable to locate solver file:", solver_file);
    return false;
  }
  
  // Load solver parameter.
  caffe::SolverParameter solver_param;
  caffe::ReadSolverParamsFromTextFileOrDie(solver_file, &solver_param);
  
  // Using GPUs.
  std::vector<int> gpus;
  int count = 0;
  // Get the number of cuda devices.
  CUDA_CHECK(cudaGetDeviceCount(&count));
  if (count == 0) {
    ErrMsg(__func__, "Training requires at least one GPU.");
    return false;
  } else {
    cudaDeviceProp device_prop;
    for (int i = 0; i < count; ++i) {
      // Display GPU information.
      cudaGetDeviceProperties(&device_prop, i);
      std::cout << "GPU " << i << ": " << device_prop.name;
      // Add gpu only when the number is matched with the user defined number.
      if(param_.device_gpu_num_ == i) {
        gpus.push_back(i);
        std::cout << " <--- USED" << std::endl;
      } else {
        std::cout << " <--- NOT USED" << std::endl;
      }
    }
  }

  solver_param.set_device_id(gpus[0]);
  caffe::Caffe::SetDevice(gpus[0]);
  caffe::Caffe::set_mode(caffe::Caffe::GPU);
  caffe::Caffe::set_solver_count(gpus.size());

  // Create solver.
  boost::shared_ptr<caffe::Solver<float> > 
  //std::shared_ptr<caffe::Solver<float> > 
    solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));

  // If snapshot file is available, use it.  
  if(snapshot_file != "") {
    std::cout << "Loading solverstate file: " << snapshot_file << std::endl;
    if(FileExist(snapshot_file)) {
      std::cout << "Resuming from " << snapshot_file << std::endl;
      solver->Restore(snapshot_file.c_str());
    } else {
      ErrMsg(__func__, "Snapshot file was not found.");
      return false;
    }
  // If there is NO snapshot and pretrained weights are available, use them. 
  } else if(pretrained_file != "") {
    std::cout << "Loading pretrained file: " << pretrained_file << std::endl;
    if(FileExist(pretrained_file)) {
      // More than two pre-trained files can be loaded.
      std::vector<std::string> model_names;
      boost::split(model_names, pretrained_file, boost::is_any_of(",") );

      for (int i = 0; i < model_names.size(); ++i) {
        std::cout << "Finetuning from " << model_names[i] << std::endl;
        solver->net()->CopyTrainedLayersFrom(model_names[i]);
        for (int j = 0; j < solver->test_nets().size(); ++j) {
          solver->test_nets()[j]->CopyTrainedLayersFrom(model_names[i]);
        }
      }
    } else {
      ErrMsg(__func__, "Pretrained file was not found.");
      return false;
    }
  }
  

  // Call solving function in solver.cpp.
#ifdef USE_NCCL
  std::cout << "[NCCL defined] Using " << gpus.size() << " GPUs" << std::endl;
  if (gpus.size() > 1) {    
    caffe::NCCL<float> nccl(solver);
    nccl.Run(gpus, snapshot_file.size() > 0 ? snapshot_file.c_str() : nullptr);
  } else {
    solver->Solve();
  }
#else
  std::cout << "[NCCL NOT defined]" << std::endl;
  solver->Solve();
#endif
  
  return true;
}
    
// Method loading a model - model file, training file, and label file.
bool VehicleDetector::LoadModelByName(const std::string& model_name,
                                      const std::string& trained_file) {
  // Get index of backbone.
  unsigned int index_model;
  if(!param_.GetBackboneNum(model_name, index_model)) {
    ErrMsg(__func__, "Invalid backbone name", model_name);
    return false;
  }
  
  return LoadModelByFile(param_.file_net_test_[index_model], trained_file);
}
  
// Method loading a model - model file, training file, and label file.
bool VehicleDetector::LoadModelByFile(const std::string& model_file,
                                      const std::string& trained_file) {
  // File check - config.
  if(!FileExist(param_.file_net_config_)) {
    ErrMsg(__func__, "Unable to locate configuration file:", param_.file_net_config_);
    return false;
  }
  // Load and print configuration.
  caffe::Frcnn::FrcnnParam::load_param(param_.file_net_config_); 
  
  // File check - model.
  if(!FileExist(model_file)) {
    ErrMsg(__func__, "Unable to locate test model file:", model_file);
    return false;
  }
  // File check - training values.
  if(!FileExist(trained_file)) {
    ErrMsg(__func__, "Unable to locate trained file:", trained_file);
    return false;
  }

  // Set caffe mode.
  caffe::Caffe::set_mode(caffe::Caffe::GPU);

  // Load network.
  net_.reset(new caffe::Net<float>(model_file, caffe::TEST));

  // Set number of input/output.
  SetNumInputFrcnn(net_->num_inputs());
  SetNumOutputFrcnn(net_->num_outputs());
  
  // Load training weights.
  net_->CopyTrainedLayersFrom(trained_file);
  caffe::Blob<float>* input_layer = net_->input_blobs()[0];
  
  // Assign number of input channels.
  num_channels_ = input_layer->channels();

  // Exception check - number of channel.
  if (num_channels_ != kNum_Ch_RGB) {
    std::cerr << "LoadModel(): Input layer should have " << kNum_Ch_RGB << " channels." << std::endl;
    return false;
  }

  // Assign input geometry.
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  // Load labels.
  std::ifstream labels(param_.file_net_label_.c_str());

  // File check - labels.
  if(!labels) {
    ErrMsg(__func__, "Unable to open labels file:", param_.file_net_label_);
    return false;
  }
  std::string line;
  labels_.clear();
  while (std::getline(labels, line)) {
    labels_.push_back(std::string(line));
  }

  // Exception check - Size of labels.
  caffe::Blob<float>* output_layer = net_->output_blobs()[1];
  if(labels_.size() != output_layer->channels()) {
    ErrMsg(__func__, "Number of labels is different from the output layer dimension.");
    return false;
  }

  // Exception check - Bbox.
  caffe::Blob<float>* bbox_layer = net_->output_blobs()[0];
  if(labels_.size()*4 != bbox_layer->channels()) {
    ErrMsg(__func__, "Number of labels is different from the output layer dimension.");
    return false;
  }

  if(!SetMean(num_channels_,
              caffe::Frcnn::FrcnnParam::pixel_means[kCh_0],
              caffe::Frcnn::FrcnnParam::pixel_means[kCh_1],
              caffe::Frcnn::FrcnnParam::pixel_means[kCh_2])) {
    ErrMsg(__func__, "Set mean failed.");
    return false;
  }
    
  return true;
}

// Method to load mean value of images from values.
bool VehicleDetector::SetMean(const unsigned short num_ch, const float ch0, const float ch1, const float ch2) {
  // Exception check - number of channels.
  if(num_ch != num_channels_) {
    std::cerr << __func__ << "(): Number of channels (" << num_ch << ") doesn't match input layer (" << num_channels_ << ")." << std::endl;
    return false;
  }
	
  // Now the number of channels is either 1 or 3.
  // Set mean value.
  if(num_ch == kNum_Ch_Grey) {
    mean_[kCh_0] = ch0;
  } else if(num_ch == kNum_Ch_RGB) {
    mean_[kCh_0] = ch0;
    mean_[kCh_1] = ch1;
    mean_[kCh_2] = ch2;
  } else {
    // Error.
    std::cerr << __func__ << "(): Number of channels (" << num_ch << ") should be " << kNum_Ch_Grey << " or " << kNum_Ch_RGB << "." << std::endl;
    return false;
  }

  return true;
}

// Method to preprocess the input image.
bool VehicleDetector::Preprocess(const cv::Mat img, ENUM_IMAGE_NORMALIZE mode,
                                 float *img_info) {
  // Get image size.
  const unsigned int height = img.rows;
  const unsigned int width = img.cols;
  
  // Exception check - image size.
  if (height == 0 || width == 0) {
    ErrMsg(__func__, "Invalid image size.");
    return false;
  }

  // Create new image for mean normalization.
  cv::Mat img_new(height, width, CV_32FC3, cv::Scalar(0, 0, 0));

  // Mean normalization by using preset mean values.
  float norm_factor = 0.;
  switch(mode) {
    // Use preset mean values.
    case kImg_Norm_Subtract_Mean:
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          img_new.at<cv::Vec3f>(cv::Point(w, h))[kCh_0] = 
            float(img.at<cv::Vec3b>(cv::Point(w, h))[kCh_0]) - mean_[kCh_0];
          img_new.at<cv::Vec3f>(cv::Point(w, h))[kCh_1] = 
            float(img.at<cv::Vec3b>(cv::Point(w, h))[kCh_1]) - mean_[kCh_1];
          img_new.at<cv::Vec3f>(cv::Point(w, h))[kCh_2] = 
            float(img.at<cv::Vec3b>(cv::Point(w, h))[kCh_2]) - mean_[kCh_2];    
        }
      }
      break;
    case kImg_Norm_Subtract_Divide_128:
      norm_factor = 128.;
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          img_new.at<cv::Vec3f>(cv::Point(w, h))[kCh_0] = 
            float(img.at<cv::Vec3b>(cv::Point(w, h))[kCh_0]) - norm_factor;
          img_new.at<cv::Vec3f>(cv::Point(w, h))[kCh_1] = 
            float(img.at<cv::Vec3b>(cv::Point(w, h))[kCh_1]) - norm_factor;
          img_new.at<cv::Vec3f>(cv::Point(w, h))[kCh_2] = 
            float(img.at<cv::Vec3b>(cv::Point(w, h))[kCh_2]) - norm_factor;  
          img_new.at<cv::Vec3f>(cv::Point(w, h))[kCh_0] /= norm_factor;
          img_new.at<cv::Vec3f>(cv::Point(w, h))[kCh_1] /= norm_factor;
          img_new.at<cv::Vec3f>(cv::Point(w, h))[kCh_2] /= norm_factor;      
        }
      }
      break;
    case kImg_Norm_Divide_256:
      norm_factor = 255.;
      img.convertTo(img_new, CV_32FC3, 1.f/norm_factor);
      break;
  }

  // Max image size comparation to know if resize is needed.
  const int max_side = MAX(height, width);
  const int min_side = MIN(height, width);
  // Find scales of max, min side.
  const float max_side_scale = float(max_side) / caffe::Frcnn::FrcnnParam::max_size ;
  const float min_side_scale = float(min_side) / caffe::Frcnn::FrcnnParam::scales[0];
  // Find the max scale.
  const float max_scale = MAX(max_side_scale, min_side_scale);
  
  // Adjust image scale.
  float img_scale = 1.0;
  if(max_scale > 1) {
    img_scale = 1.0 / max_scale;	
  }
  
  // Calculate resized height and width. 
  const unsigned int height_resized = int(height * img_scale);
  const unsigned int width_resized = int(width * img_scale);

  // Create resized image and data buffer.
  cv::Mat img_resized;
  cv::resize(img_new, img_resized, cv::Size(width_resized, height_resized));
  float data_buf[height_resized * width_resized * kNum_Ch_RGB];

  // Copy resized image to data buffer.
  for (int h = 0; h < height_resized; ++h) {
    for (int w = 0; w < width_resized; ++w) {
      data_buf[(kCh_0 * height_resized + h) * width_resized + w] = float(img_resized.at<cv::Vec3f>(cv::Point(w, h))[kCh_0]);
      data_buf[(kCh_1 * height_resized + h) * width_resized + w] = float(img_resized.at<cv::Vec3f>(cv::Point(w, h))[kCh_1]);
      data_buf[(kCh_2 * height_resized + h) * width_resized + w] = float(img_resized.at<cv::Vec3f>(cv::Point(w, h))[kCh_2]);
    }
  }

  // Reshape data input.
  net_->blob_by_name("data")->Reshape(1, num_channels_, height_resized, width_resized);
  caffe::Blob<float> * input_blobs= net_->input_blobs()[0];

  caffe::caffe_gpu_memcpy(sizeof(float)* input_blobs->count(), data_buf, input_blobs->mutable_gpu_data());

  // Store resize values and scale.
  img_info[kImg_Resized_Height] = height_resized;
  img_info[kImg_Resized_Width] = width_resized;
  img_info[kImg_Resized_Scale] = img_scale;

  net_->blob_by_name("im_info")->set_cpu_data(img_info);
	
  return true;
}

// Detection process from the input image.
bool VehicleDetector::Detect(const cv::Mat img, const bool debug_msg,
                             std::vector<Vehicle>& detected_vehicles) {
  // Check if input image exists.
  if(img.empty()) {
    ErrMsg(__func__, "Can not reach the input image.");
    return false;
  }

  // Get image size.
  const unsigned int height = img.rows;
  const unsigned int width = img.cols;
  
  // It is necessary to pass as pointers in order to keep them in the net.
  float *img_info = new float[kImg_Resized_Info_Size];
  // Preprocess input image - error message is printed in the function.
  if(!Preprocess(img, kImg_Norm_Subtract_Mean, img_info)) {
    delete [] img_info;
    return false;
  }

  net_->ForwardFrom(0);

  // Extract results from the network below.
	
  // Get number of classes.
  const unsigned int num_classes = labels_.size();
  // Get predicted bboxes from the network.
  const float* bbox_delta = net_->blob_by_name("bbox_pred")->cpu_data();
  // Get probability for each class.
  const float* pred_cls = net_->blob_by_name("cls_prob")->cpu_data(); 
  // Get RoIs and the number of RoIs from the network.
  const float* rois = net_->blob_by_name("rois")->cpu_data(); 		
  const unsigned int num_rois = net_->blob_by_name("rois")->num();
  
  // Exception check.
  if(bbox_delta == nullptr || pred_cls == nullptr || rois == nullptr) {
    ErrMsg(__func__, "Invalid data was returned from the network.");
    delete [] img_info;
    return false;
  }
    
  // Get bbox of RoI at the original input image by dividing the scale.
  float *boxes = new float[num_rois * kNum_Data_Bbox];
  for (int n = 0; n < num_rois; n++) {
    for (int c = 0; c < kNum_Data_Bbox; c++) {
      boxes[n * kNum_Data_Bbox + c] = rois[n * kNum_Data_Pred + c + 1] / img_info[kImg_Resized_Scale];
    }
  }
  delete [] img_info;
  
  // All prediction of bbox and RoI are stored below.
  float *pred = new float[num_rois * kNum_Data_Pred * num_classes];
  
  // Calculate bbox region at the input image.
  InvTransformBbox(num_rois, num_classes, bbox_delta, pred_cls, boxes,
      pred, height, width);
      
  // Release memory.
  delete []boxes;
  
  // Predictions are sorted in each class.
  float **pred_per_class = new float*[num_classes];
  float **sorted_pred_cls = new float*[num_classes];
  int **keep = new int*[num_classes];
  for (int i = 0; i < num_classes; i++) {
    pred_per_class[i] = new float[num_rois * kNum_Data_Pred];
    sorted_pred_cls[i] = new float[num_rois * kNum_Data_Pred];
    keep[i] = new int[num_rois];
  }
  
  int *num_keep = new int[num_classes];
  // Get threshold for non-maximum suppression, from configuration.
  const float nms_threshold = caffe::Frcnn::FrcnnParam::test_nms;
  
  // Background class is ignored below.
  const unsigned int start_index_without_bg = 1;
  
  // Collect all RoIs.
  for (int j = 0; j < num_rois; j++) {
    for (int k = 0; k < kNum_Data_Pred; k++) {
      pred_per_class[kLabel_Car][j * kNum_Data_Pred + k] = pred[(kLabel_Car * num_rois + j) * kNum_Data_Pred + k];
    } 
  }
  
  
  // non-maximum surpression(NMS).
  // The FRCNN version of the caffe provides NMS function, but it misses some ovelapped box.
  // Therefore, NMS is implemented here.
  // Also, Soft-NMS (score = score * (1 - iou)) didn't show the better result for the vehicle detection.
  
  // Sort Bbox within a class.
  SortBbox(num_rois, pred_per_class[kLabel_Car], sorted_pred_cls[kLabel_Car]);
  
  // NMS.
  for (int i = 0; i < num_rois - 1; i++) {
    float score = sorted_pred_cls[kLabel_Car][i*kNum_Data_Pred + kROI_Score];
    if(score <= 0.) {
        continue;
    }
    float x, y, width, height;
    // Get first box.
    x = sorted_pred_cls[kLabel_Car][i*kNum_Data_Pred + kROI_Left];
    y = sorted_pred_cls[kLabel_Car][i*kNum_Data_Pred + kROI_Top];
    width = sorted_pred_cls[kLabel_Car][i*kNum_Data_Pred +  kROI_Right] - x;
    height = sorted_pred_cls[kLabel_Car][i*kNum_Data_Pred +  kROI_Bottom] - y;

    if(width <= 0 || height <= 0) {
      continue;
    }
    cv::Rect_<float> box_first = cv::Rect_<float>(x, y, width, height);  

    for (int j = i + 1; j < num_rois; j++) {
      score = sorted_pred_cls[kLabel_Car][j*kNum_Data_Pred + kROI_Score];
      if(score <= 0.) {
        continue;
      }
      
      // Get second box.
      x = sorted_pred_cls[kLabel_Car][j*kNum_Data_Pred + kROI_Left];
      y = sorted_pred_cls[kLabel_Car][j*kNum_Data_Pred + kROI_Top];
      width = sorted_pred_cls[kLabel_Car][j*kNum_Data_Pred +  kROI_Right] - x;
      height = sorted_pred_cls[kLabel_Car][j*kNum_Data_Pred +  kROI_Bottom] - y;

      if(width > 0 && height > 0) {
        cv::Rect_<float> box_second = cv::Rect_<float>(x, y, width, height);
        const float iou = CalculateIoU(box_first, box_second);
        // Set score as 0 if IoU is greater than the threshold.
        if(iou > nms_threshold) {
          sorted_pred_cls[kLabel_Car][j*kNum_Data_Pred + kROI_Score] = 0.;
        }
      }
    }
  }
  
  unsigned int count = 0;
  for (int i = 0; i < num_rois - 1; i++) {
    const float score = sorted_pred_cls[kLabel_Car][i*kNum_Data_Pred + kROI_Score];
    if(score > 0.) {
      keep[kLabel_Car][count] = i;
      count ++;
    }
  }
  num_keep[kLabel_Car] = count;
  
  // Get lower threshold to determine which object to detect, from configuration.
  float min_threshold = caffe::Frcnn::FrcnnParam::test_score_thresh;
  if(debug_msg) {
    std::cout << "Score threshold: " << min_threshold << std::endl;
  }
  // For each class, generate results within the number to keep where the score is above the threshold.
  int k = 0;
  while (k < num_keep[kLabel_Car]) {    
    if(sorted_pred_cls[kLabel_Car][keep[kLabel_Car][k] * kNum_Data_Pred + kROI_Score] > min_threshold) {
      float x = sorted_pred_cls[kLabel_Car][keep[kLabel_Car][k] * kNum_Data_Pred + kROI_Left];
      float y = sorted_pred_cls[kLabel_Car][keep[kLabel_Car][k] * kNum_Data_Pred + kROI_Top];
      float width = sorted_pred_cls[kLabel_Car][keep[kLabel_Car][k] * kNum_Data_Pred + kROI_Right] - x;
      float height = sorted_pred_cls[kLabel_Car][keep[kLabel_Car][k] * kNum_Data_Pred + kROI_Bottom] - y;
      float score = sorted_pred_cls[kLabel_Car][keep[kLabel_Car][k] * kNum_Data_Pred + kROI_Score];

      // Print detections.
      if(debug_msg) {
        std::cout << "Detected object: " << labels_[kLabel_Car] << " Score: " << score;
        std::cout << " Box: " << x << " " << y << " " << width << " " << height << std::endl;
      }

      // Assign detections.
      Vehicle aux(score, cv::Rect_<float>(x, y, width, height));
      detected_vehicles.push_back(aux);
    }
    k++;
  }

  // Release memories.
  delete[] pred;
  for (int i = 0; i < num_classes; i++) {
    delete[] pred_per_class[i];
    delete[] sorted_pred_cls[i];
    delete[] keep[i];
  }
  delete[] pred_per_class;
  delete[] sorted_pred_cls;
  delete[] keep;
    
  return true;
}

// Sort RoIs by score.
void VehicleDetector::SortBbox(const unsigned int num_rois, const float* pred, float* sorted_pred)
{
  std::vector<PredictionInfo> pred_info;
  PredictionInfo tmp;
  for (int i = 0; i < num_rois; i++) {
    tmp.score_ = pred[i * kNum_Data_Pred + kROI_Score];
    tmp.head_ = pred + i * kNum_Data_Pred;
    pred_info.push_back(tmp);
  }

  // Sort scores by descending order.
  std::sort(pred_info.begin(), pred_info.end());
  
  for (int i = 0; i < num_rois; i++) {
    for (int j = 0; j < kNum_Data_Pred; j++) {
      sorted_pred[i * kNum_Data_Pred + j] = pred_info[i].head_[j];
    }
  }
}

// Method for inverse transform of bbox.
void VehicleDetector::InvTransformBbox(const unsigned int num_rois, 
    const unsigned int num_classes, const float* box_deltas, const float* pred_cls,
    float* boxes, float* pred, const unsigned int img_height, const unsigned int img_width) {
  float width, height;
  float center_x, center_y;
  float dx, dy;
  float dw, dh; 
  float pred_center_x, pred_center_y;
  float pred_w, pred_h;
  
  for(int i = 0; i < num_rois; i++) {
    // For each RoI, calculate center and size of bbox.
    width = boxes[i * kNum_Data_Bbox + kROI_Right] - boxes[i * kNum_Data_Bbox + kROI_Left] + 1.0;
    height = boxes[i * kNum_Data_Bbox + kROI_Bottom] - boxes[i * kNum_Data_Bbox + kROI_Top] + 1.0;
    center_x = boxes[i * kNum_Data_Bbox + kROI_Left] + 0.5 * width;
    center_y = boxes[i * kNum_Data_Bbox + kROI_Top] + 0.5 * height;
   
    for (int j = 0; j < num_classes; j++) {
      // Compute deltas, predicted bbox, and assign bbox and score of label.
      dx = box_deltas[(i * num_classes + j) * kNum_Data_Bbox + kROI_Left];
      dy = box_deltas[(i * num_classes + j) * kNum_Data_Bbox + kROI_Top];
      dw = box_deltas[(i * num_classes + j) * kNum_Data_Bbox + kROI_Width];
      dh = box_deltas[(i * num_classes + j) * kNum_Data_Bbox + kROI_Height];

      pred_center_x = center_x + width * dx;
      pred_center_y = center_y + height * dy;
      pred_w = width * exp(dw);
      pred_h = height * exp(dh);
      
      pred[(j * num_rois + i) * kNum_Data_Pred + kROI_Left] = MAX(MIN(pred_center_x - 0.5 * pred_w, img_width -1), 0);
      pred[(j * num_rois + i) * kNum_Data_Pred + kROI_Top] = MAX(MIN(pred_center_y - 0.5 * pred_h, img_height -1), 0);
      pred[(j * num_rois + i) * kNum_Data_Pred + kROI_Right] = MAX(MIN(pred_center_x + 0.5 * pred_w, img_width -1), 0);
      pred[(j * num_rois + i) * kNum_Data_Pred + kROI_Bottom] = MAX(MIN(pred_center_y + 0.5 * pred_h, img_height -1), 0);
      pred[(j * num_rois + i) * kNum_Data_Pred + kROI_Score] = pred_cls[i*num_classes+j];
    }
  }
}

// Visualize detection data on the image.
void VehicleDetector::VisualizeDetections(cv::Mat& img,
                                          std::vector<Vehicle> vehicles,
                                          ENUM_BBOX_TYPE type,
                                          ENUM_BBOX_INFO info) {
  for(int i = 0; i < vehicles.size(); i++) {
    cv::Rect_<float> rect = vehicles[i].GetBbox();
    
    // Choose color.
    cv::Scalar color;
    if(type == kBbox_GT) {
      color = color_red;
    } else if(type == kBbox_dontcare) {
      color = color_gray;
    } else {
      color = color_blue;
    }
    // Draw bbox.
    cv::rectangle(img, cv::Point(rect.x, rect.y), 
                    cv::Point(rect.x + rect.width, rect.y + rect.height),
                    color, 2, 8, 0);
    
    // Write text.
    if(type == kBbox_GT || type == kBbox_pred) {
      std::string text;
      if(info == kBbox_Score && type == kBbox_pred) {
        // Write score.
        text = std::to_string(vehicles[i].GetScore());     
      } else {
        // Write coordinates.
        std::stringstream stream_pos_x, stream_pos_z;
        stream_pos_x << std::fixed << std::setprecision(2) 
          << vehicles[i].Get3DLoc(k3D_X);
        stream_pos_z << std::fixed << std::setprecision(2) 
          << vehicles[i].Get3DLoc(k3D_Z);
        text = "X:" + stream_pos_x.str() + " Z:" + stream_pos_z.str();
      }
      
      WriteText(img, rect.x, rect.y, text, param_.font_scale_, param_.font_thickness_);
    }
  }
}
  
// Extract features from network and save.
void VehicleDetector::ConvFeatures(const std::string& layer_name, 
                                   std::vector<cv::Mat>& img_feat) {
  if(net_->blob_by_name(layer_name) == 0) {
    ErrMsg(__func__, "Unknown blob name:", layer_name);
    return;
  }
  
  // Print layer info.
  std::cout << layer_name << " shape: " 
    << net_->blob_by_name(layer_name)->shape_string() << std::endl;

  // Get layer shape.
  std::vector<int> shape = net_->blob_by_name(layer_name)->shape();
  if(shape.size() == 0) {
    ErrMsg(__func__, "Layer shape is invalid.");
    return;
  }
  
  const unsigned int width = shape[kShape_Width];
  const unsigned int height = shape[kShape_Height];
  const unsigned int step = width * height;
  
  // size check.
  if(width == 0 || height == 0) {
    ErrMsg(__func__, "Size is invalid.");
    return;
  }
  // Get raw data.
  const float* data = net_->blob_by_name(layer_name)->cpu_data();
  if(data == nullptr) {
    ErrMsg(__func__, "Data is invalid.");
    return;
  }
  
  // Copy data.
  cv::Mat feat(height, width, CV_32FC1, cv::Scalar(0));
  for (int i = 0; i < shape[kShape_Kernel]; ++i) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        int idx = (i * step) + width * h + w;
        feat.at<float>(cv::Point(w, h)) = data[idx];
      }
    }
    cv::normalize(feat, feat, 0, 255, cv::NORM_MINMAX);
    img_feat.push_back(feat.clone());
  }
}
  
void VehicleDetector::SaveAllConvolutionImages(const std::string& path) {
  unsigned int count_save = 0;
  for(int i = 0; i < net_->layers().size(); ++i) {
    // Match layer type's name.
    const std::string layer_type = net_->layers()[i]->type();
    if(!layer_type.compare("Convolution") || !layer_type.compare("Pooling")) {
      count_save++;
      // Get layer name.
      const std::string layer_name = net_->layer_names()[i];
      std::cout << "[" << layer_type << "] " << layer_name << std::endl;
      // Get an image from blob by using the layer name.
      std::vector<cv::Mat> img;
      ConvFeatures(layer_name, img);
      
      if(img.size() > 0) {
        // Get image width.
        unsigned int width = img[0].cols;
        // Number of columns for the merged image.
        const unsigned int cols_default = param_.save_image_conv_num_col_merged_from_;
        unsigned int cols = cols_default;
        while(width * cols < param_.save_image_conv_large_width_ * cols_default
          && cols < param_.save_image_conv_num_col_merged_to_
          && cols < (img.size() / cols)) {
          cols++;
        }
        
        // Calculate number of rows.
        unsigned int rows = img.size() / cols;
        if(img.size()%cols != 0) {
          rows++;
        }
        // Create merged image.
        cv::Mat img_merged(img[0].rows * rows, img[0].cols * cols, CV_32FC1, cv::Scalar(0));
           
        // Save single image.
        const std::string path_save = path + PathSeparator() 
          + std::to_string(count_save) + "_" + layer_name + PathSeparator();
        mkdir(path_save.c_str(), 0777); 
        
        for(int j = 0; j < img.size(); ++j) {
          cv::imwrite(path_save + layer_name + "_" + std::to_string(j) + ".png", img[j]);
          img[j].copyTo(img_merged(cv::Rect((j%cols) * img[j].cols,
                                            (int)(j/cols) * img[j].rows,
                                            img[j].cols, img[j].rows)));
        }
        
        // Save merged image.
        cv::imwrite(path + PathSeparator() + layer_name + ".png", img_merged);
      }      
    }
  }
}
  
// Calculate intersection over union (IoU).
float VehicleDetector::CalculateIoU(cv::Rect_<float> bbox_pred, cv::Rect_<float> bbox,
                                    ENUM_BBOX_TYPE box_type) {
  // Exception.
  if(bbox_pred.width <= 0. || bbox_pred.height <= 0. || bbox.width <= 0. || bbox.height <= 0.) {
    std::cerr << __func__ << "(): Invalid bbox."
      << "[Pred]w:" << bbox_pred.width << " h:" << bbox_pred.height
      << "[GT/DontCare]w:" << bbox.width << " h:" << bbox.height
      << std::endl;
    return 0;
  }
  
  // Intersection area.
  const float x_min = cv::max(bbox_pred.x, bbox.x);
  const float y_min = cv::max(bbox_pred.y, bbox.y);
  const float x_max = cv::min(bbox_pred.x + bbox_pred.width, bbox.x + bbox.width);
  const float y_max = cv::min(bbox_pred.y + bbox_pred.height, bbox.y + bbox.height);
  const float width = cv::max(x_max - x_min + 0.0, 0.0);
  const float height = cv::max(y_max - y_min + 0.0, 0.0);
  const float intersection = width * height;
  
  // Union.
  const float area_bbox_pred = bbox_pred.width * bbox_pred.height;
  const float area_bbox = bbox.width * bbox.height;
  
  float iou = 0.;
  if(box_type == kBbox_GT) {
    float uni = area_bbox_pred + area_bbox - intersection;
    // Prevent dividing by 0.
    // Also union must be greater than 0.
    if(uni <= 0) {
      ErrMsg(__func__, "Union area must be greater than 0.");
      return 0;
    }
    iou = intersection / uni;
  } else if(box_type == kBbox_dontcare) {
    iou = intersection / area_bbox_pred;
  }
  
  return iou;
} 
  

} // namespace vehicle_detector
} // namespace robotics
