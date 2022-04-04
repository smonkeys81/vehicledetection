// Copyright © 2020 Robotics, Inc. All Rights Reserved.

// This file provides units tests for Dataset class.

#include "dataset.h"
#include "gtest/gtest.h"
//#include <opencv2/highgui/highgui.hpp>


namespace robotics {
namespace vehicle_detector {


/// \class DataSetTest
/// This class is for the unittest.
class DataSetTest : public ::testing::Test {
protected:
  /// \brief Constructor.
  DataSetTest() {
    // Create test image.
    img_test = cv::Mat(width, height, CV_16S, cv::Scalar(0));
    // Make half of the image white.
    for(int row = 0; row < height/2; ++row) {
      for(int col = 0; col < width; ++col) {
        img_test.at<short>(row, col) = 255;
      }
    }
    cv::imwrite("test_img.png", img_test);
  }

  /// \brief Destructor.
  virtual ~DataSetTest() {}
	
  /// \brief This method is called immediately after the constructor.
  virtual void SetUp() {}

  /// \brief This method is called immediately after each test, right before the destructor.
  virtual void TearDown() {}	

protected:
  /// \brief DataSet instance.
  DataSet data_;

  /// \brief Test image size - width.
  const unsigned int width = 200;
  
  /// \brief Test image size - height.
  const unsigned int height = 200;
  
  /// \brief Test image.
  cv::Mat img_test;
  
  /// \brief Test horizontal FOV;
  const float h_fov_deg = 90;
};


// Test CalculateAvgIntensity function.
// Check if the white box returns 255.
TEST_F(DataSetTest, CalculateAvgIntensity_white) {
  // Set bbox in white region.
  const cv::Rect_<float> box(0, 0, 200, 100);
  float result = data_.CalculateAvgIntensity(img_test, box);
  ASSERT_EQ(result, 255) << "Return value from the method is not matched with the expection.";
}
  
// Test CalculateAvgIntensity function.
// Check if the black box returns 0.
TEST_F(DataSetTest, CalculateAvgIntensity_black) {
  // Set bbox in black region.
  const cv::Rect_<float> box(0, 100, 200, 100);
  float result = data_.CalculateAvgIntensity(img_test, box);
  ASSERT_EQ(result, 0) << "Return value from the method is not matched with the expection.";
}
  
// Test CalculateAvgIntensity function.
// Check if the entire box returns 255, ignoring pixels with zero value.
TEST_F(DataSetTest, CalculateAvgIntensity_allregion) {
  // Set bbox in all region.
  // Zero value is not counted in the original design.
  const cv::Rect_<float> box(0, 0, 200, 200);
  float result = data_.CalculateAvgIntensity(img_test, box);
  ASSERT_EQ(result, 255) << "Return value from the method is not matched with the expection.";
}
  
// Test CalculateAvgIntensity function.
// Check if the box region out of the image is ignored.
// The region out of the image may have garbage values.
// Case: Left and Top positions are less than 0.
TEST_F(DataSetTest, CalculateAvgIntensity_outofboundary1) {
  // Set bbox out of boundary.
  // Zero value is not counted in the original design.
  const cv::Rect_<float> box(-100, -100, 200, 200);
  float result = data_.CalculateAvgIntensity(img_test, box);
  ASSERT_EQ(result, 255) << "Return value from the method is not matched with the expection.";
}

// Test CalculateAvgIntensity function.
// Check if the box region out of the image is ignored.
// The region out of the image may have garbage values.
// Case: Right and Bottom positions are bigger than the image boundary.
TEST_F(DataSetTest, CalculateAvgIntensity_outofboundary2) {
  // Set bbox out of boundary.
  // Zero value is not counted in the original design.
  const cv::Rect_<float> box(100, 100, 200, 200);
  float result = data_.CalculateAvgIntensity(img_test, box);
  ASSERT_EQ(result, 0) << "Return value from the method is not matched with the expection.";
}
  
// Test CalculateAngle function.
// Check if the function returns 0 when the center of box is exactly in the middle of the image.
TEST_F(DataSetTest, CalculateAngle_center) {
  const cv::Rect_<float> box(50, 0, 100, 100);
  float result = data_.CalculateAngle(h_fov_deg, width, box);
  ASSERT_EQ(result, 0) << "Return value from the method is not matched with the expection.";
}

// Test CalculateAngle function.
// Check if the function returns the left most angle when the center of box is 0 in column.
TEST_F(DataSetTest, CalculateAngle_left) {
  cv::Rect_<float> box(-50, 0, 100, 100);
  float result = data_.CalculateAngle(h_fov_deg, width, box);
  ASSERT_EQ(result, -h_fov_deg/2) << "Return value from the method is not matched with the expection.";
}
  
// Test CalculateAngle function.
// Check if the function returns the right most angle when the center of box is the same as the column size of the image.
TEST_F(DataSetTest, CalculateAngle_right) {
  cv::Rect_<float> box(150, 0, 100, 100);
  float result = data_.CalculateAngle(h_fov_deg, width, box);
  ASSERT_EQ(result, h_fov_deg/2) << "Return value from the method is not matched with the expection.";
}
  
  
} // namespace vehicle_detector
} // namespace robotics
