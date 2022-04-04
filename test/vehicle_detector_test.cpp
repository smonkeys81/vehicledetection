// Copyright © 2019 Robotics, Inc. All Rights Reserved.

// This file provides units tests for VehicleDetector class.

#include "vehicle_detector.h"
#include "gtest/gtest.h"
#include <opencv2/highgui/highgui.hpp>


namespace robotics {
namespace vehicle_detector {


/// \class VehicleDetectorTest
/// This class is for the unittest.
class VehicleDetectorTest : public ::testing::Test {
protected:
  /// \brief Constructor.
  VehicleDetectorTest() {
    detector_ = new VehicleDetector(config_);
    num_rois_ = 2; // for simple test.
    pred_ = new float[kNum_Data_Pred * num_rois_];
    sorted_pred_ = new float[kNum_Data_Pred * num_rois_];
  }

  /// \brief Destructor.
  virtual ~VehicleDetectorTest() {
    delete[] sorted_pred_;
    delete[] pred_;
    delete detector_;
  }
	
  /// \brief This method is called immediately after the constructor.
  virtual void SetUp() {}

  /// \brief This method is called immediately after each test, right before the destructor.
  virtual void TearDown() {}	
		
  /// \brief Set number of channels in vehicle detector for test.
  /// \param[in] detector Instance of VehicleDetector class.
  /// \param[in] num_channels Number of channels to set.
  void SetChannels(VehicleDetector* detector, const int num_channels) { 
    detector->num_channels_ = num_channels; 
  }

  /// \brief Call Preprocess function.
  /// \param[in] detector Instance of VehicleDetector class.
  /// \param[in] img Input image.
  /// \param[out] img_info Height, width, and scale of the image.
  /// \return True if preprocess is done successfully, false otherwise.
  bool Preprocess(VehicleDetector* detector, cv::Mat img, float *img_info) { 
    return detector->Preprocess(img, kImg_Norm_Subtract_Mean, img_info); 
  }
  
  /// \brief Call SortBbox function.
  /// \param[in] detector Instance of VehicleDetector class.
  /// \param[in] num_rois Number of RoIs.
  /// \param[in] pred Predicted data consists of bboxes and scores.
  /// \param[out] sorted_pred Sorted prediction data by score.
  void SortBbox(VehicleDetector* detector, const unsigned int num_rois, const float* pred, float* sorted_pred) {
    detector->SortBbox(num_rois, pred, sorted_pred);
  }

protected:
  /// \brief VehicleDetector instance.
  VehicleDetector *detector_;

  /// \brief Test output directory.
  const std::string out_dir_ = GetTestOutputDir();

  /// \brief File path to the configuration.
  const std::string config_ = out_dir_ + "/../config/config.json";

  /// \brief File path to solver.
  const std::string solver_ = out_dir_ + "/sample_model/ZF_faster_rcnn_solver_test.pt";
  
  /// \brief File path to the pretrained weights.
  const std::string pretrain_ = out_dir_ + "/sample_model/ZF.v2.caffemodel";

  /// \brief File path to the model.
  const std::string model_ = out_dir_ + "/sample_model/ZF_faster_rcnn_test.pt";

  /// \brief File path to the trained weights.
  const std::string train_ = out_dir_ + "/sample_model/ZF_faster_rcnn_final.caffemodel";

  /// \brief File name of the test image.
  const std::string test_img_file_ = "000008.png";
  
  /// \brief File path to the test image.
  const std::string test_img_ = out_dir_ + "/sample_img/" + test_img_file_;
  
  /// \brief Arbitrary mean value for test.
  const float mean_val_ = 128.;
  
  /// \brief Number of RoIs.
  unsigned int num_rois_;
  
  /// \brief Container for predictions.
  float* pred_;
  
  /// \brief Container for sorted predictions from pred_.
  float* sorted_pred_;
};


// Test SetMean function.
TEST_F(VehicleDetectorTest, bbox_sort) {
  // Assign scores ascending order.
  const int low = 0;
  const int high = 1;
  pred_[kROI_Score] = low;
  pred_[kNum_Data_Pred + kROI_Score] = high;
  
  // Call the function to sort by descending order.
  SortBbox(detector_, num_rois_, pred_, sorted_pred_);
  
  ASSERT_EQ(sorted_pred_[kROI_Score], high) << "Return value from the method is not matched with the expection."; 
  ASSERT_EQ(sorted_pred_[kNum_Data_Pred + kROI_Score], low) << "Return value from the method is not matched with the expection."; 
}

// Test load_model function.
// If file is invalid, the called function in Caffe library terminate the program.
// Therefore this test only check if the return value is true when all valid files were passed.
TEST_F(VehicleDetectorTest, load_model_and_detect) {
  // Load model test.
  bool b1 = false;
  b1 = detector_->LoadModelByFile(model_, train_);
  ASSERT_TRUE(b1) << "Return value from the method is not matched with the expection."; 

  // Detection test - this cannot be splitted since detect function requires loaded model, weights, and labels.
  // Preprocess function test can be skipped as this function is called in Detect function together.
  bool b2 = false;
  cv::Mat img = cv::imread(test_img_, -1);
  std::vector<Vehicle> detections;
  b2 = detector_->Detect(img, true, detections);
  ASSERT_TRUE(b2) << "Return value from the method is not matched with the expection."; 
}

// Test SetMean function.
TEST_F(VehicleDetectorTest, set_mean) {
  // Assign 1 at num_channels_.
  SetChannels(detector_, kNum_Ch_Grey);

  bool b1 = false;
  b1 = detector_->SetMean(kNum_Ch_Grey, mean_val_);
  ASSERT_TRUE(b1) << "Return value from the method is not matched with the expection."; 

  bool b2 = true;
  b2 = detector_->SetMean(kNum_Ch_RGB, mean_val_, mean_val_ ,mean_val_);
  ASSERT_FALSE(b2) << "Return value from the method is not matched with the expection."; 

  // Assign 3 at num_channels_.
  SetChannels(detector_, kNum_Ch_RGB);

  bool b3 = true;
  b3 = detector_->SetMean(kNum_Ch_Grey, mean_val_);
  ASSERT_FALSE(b3) << "Return value from the method is not matched with the expection."; 

  bool b4 = false;
  b4 = detector_->SetMean(kNum_Ch_RGB, mean_val_, mean_val_ ,mean_val_);
  ASSERT_TRUE(b4) << "Return value from the method is not matched with the expection."; 
}

// Test Training function.
TEST_F(VehicleDetectorTest, train_model) {
  std::cout << "[Path to test files]" << std::endl;
  std::cout << "Solver: " << solver_ << std::endl;
  std::cout << "Pretrained model: " << pretrain_ << std::endl;
  
  // Call training function and check if the return value is true.
  ASSERT_TRUE(detector_->TrainModel(solver_, pretrain_)) << "Return value from the method is not matched with the expection."; 
}

// Test CalculateIoU function.
TEST_F(VehicleDetectorTest, calc_IoU1) {
  // Test case - two identical boxes.
  cv::Rect_<float> bbox_test(0, 0, 100, 100);
  
  // Call function.
  float IoU = detector_->CalculateIoU(bbox_test, bbox_test);
  // Check.
  ASSERT_NEAR(IoU, 1.0, std::numeric_limits<float>::epsilon()) << "Return value from the method is not matched with the expection."; 
}
  
// Test CalculateIoU function.
TEST_F(VehicleDetectorTest, calc_IoU2) {
  // Test case - two boxes without overlap.
  cv::Rect_<float> bbox_test1(0, 0, 100, 100);
  cv::Rect_<float> bbox_test2(100, 0, 100, 100);
  cv::Rect_<float> bbox_test3(0, 100, 100, 100);
  
  // Call function.
  float IoU1 = detector_->CalculateIoU(bbox_test1, bbox_test2);
  // Check.
  ASSERT_NEAR(IoU1, 0.0, std::numeric_limits<float>::epsilon()) << "Return value from the method is not matched with the expection."; 
  
  // Call function.
  float IoU2 = detector_->CalculateIoU(bbox_test1, bbox_test3);
  // Check.
  ASSERT_NEAR(IoU2, 0.0, std::numeric_limits<float>::epsilon()) << "Return value from the method is not matched with the expection.";
}
  
// Test CalculateIoU function.
TEST_F(VehicleDetectorTest, calc_IoU3) {
  // Test case.
  cv::Rect_<float> bbox_test1(0, 0, 100, 100);
  cv::Rect_<float> bbox_test2(50, 50, 50, 50);
  
  // Call function.
  float IoU = detector_->CalculateIoU(bbox_test1, bbox_test2);
  // Check.
  ASSERT_NEAR(IoU, 0.25, std::numeric_limits<float>::epsilon()) << "Return value from the method is not matched with the expection."; 
}
  
  
} // namespace vehicle_detector
} // namespace robotics
