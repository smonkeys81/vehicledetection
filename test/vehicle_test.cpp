// Copyright © 2019 Robotics, Inc. All Rights Reserved.

// This file provides units tests for Vehicle class.

#include "gtest/gtest.h"
#include "vehicle.h"


namespace robotics {
namespace vehicle_detector {
  
  
/// \class VehicleDetectorTest
/// This class is for the unittest.
class VehicleTest : public ::testing::Test {
protected:
  /// \brief Constructor.
  VehicleTest() {
    v_.SetScore(0.333);
    v_.SetBbox(cv::Rect_<float>(5,6,7,8)); 
  }

  /// \brief Destructor.
  virtual ~VehicleTest() {}
	
  /// \brief This method is called immediately after the constructor.
  virtual void SetUp() {}

  /// \brief This method is called immediately after each test, right before the destructor.
  virtual void TearDown() {}	

protected:
  /// \brief VehicleDetector instance.
  Vehicle v_;
};

// Test Score returning function.
TEST_F(VehicleTest, return_score) {
  float f = -1.;
  f = v_.GetScore();
  ASSERT_NEAR(0.333, f, std::numeric_limits<float>::epsilon()) << "Score returned from the method is not matched with the original input"; 
}

// Test Bbox returning function.
TEST_F(VehicleTest, return_bbox) {
  cv::Rect_<float> rect(5,6,7,8);
  cv::Rect_<float> rect_return = v_.GetBbox();
  ASSERT_EQ(rect, rect_return) << "Bbox returned from the method is not matched with the original input";
}


} // namespace vehicle_detector
} // namespace robotics