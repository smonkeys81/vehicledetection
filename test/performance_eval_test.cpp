// Copyright © 2019 Robotics, Inc. All Rights Reserved.

// This file provides units tests for PerformanceEval class.

#include "performance_eval.h"
#include "gtest/gtest.h"

namespace robotics {
namespace vehicle_detector {


/// \class PerformanceEvalTest
/// This class is for the unittest.
class PerformanceEvalTest : public ::testing::Test {
protected:
  /// \brief Constructor.
  PerformanceEvalTest() {
    eval_ = new PerformanceEval(config_);
  }

  /// \brief Destructor.
  virtual ~PerformanceEvalTest() {
    delete eval_;
  }
	
  /// \brief This method is called immediately after the constructor.
  virtual void SetUp() {}

  /// \brief This method is called immediately after each test, right before the destructor.
  virtual void TearDown() {}

protected:
  /// \brief VehicleDetector instance.
  PerformanceEval *eval_;

  /// \brief Test output directory.
  const std::string out_dir_ = GetTestOutputDir();  

  /// \brief File path to the configuration.
  const std::string config_ = out_dir_ + "../config/config.json";
};


} // namespace vehicle_detector
} // namespace robotics
