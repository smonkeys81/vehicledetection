// Copyright © 2019 Robotics, Inc. All Rights Reserved.

// This file provides units tests for CrossValidation class.

#include "cross_validation.h"
#include "gtest/gtest.h"

namespace robotics {
namespace vehicle_detector {


/// \class PerformanceEvalTest
/// This class is for the unittest.
class CrossValidationTest : public ::testing::Test {
protected:
  /// \brief Constructor.
  CrossValidationTest() {}

  /// \brief Destructor.
  virtual ~CrossValidationTest() {}
	
  /// \brief This method is called immediately after the constructor.
  virtual void SetUp() {}

  /// \brief This method is called immediately after each test, right before the destructor.
  virtual void TearDown() {}

protected:
  /// \brief CrossValidation instance.
  CrossValidation cross_val_;
};


// Test SplitData function.
TEST_F(CrossValidationTest, data_split_return) {
  // Test case - No data inside.
  std::vector<DataImageLabel> dataset;

  std::vector<int> training, validation;
  bool b1 = cross_val_.SplitData(dataset, 1, 0, training, validation);
  bool b2 = cross_val_.SplitData(dataset, 1, 1, training, validation);
  bool b3 = cross_val_.SplitData(dataset, 0, 0, training, validation);

  // Check.
  ASSERT_EQ(b1, true) << "Return value from the method is not matched with the expection."; 
  ASSERT_EQ(b2, false) << "Return value from the method is not matched with the expection.";
  ASSERT_EQ(b3, false) << "Return value from the method is not matched with the expection."; 
}
  
// Test SplitData function.
TEST_F(CrossValidationTest, data_split_0data) {
  // Test case - No data inside.
  std::vector<DataImageLabel> dataset;

  std::vector<int> training, validation;
  cross_val_.SplitData(dataset, 1, 0, training, validation);

  unsigned int size_training = training.size();
  unsigned int size_validation = validation.size();

  // Check.
  ASSERT_EQ(size_training, 0) << "Return value from the method is not matched with the expection."; 
  ASSERT_EQ(size_validation, 0) << "Return value from the method is not matched with the expection."; 
}

// Test SplitData function.
TEST_F(CrossValidationTest, data_split_2data) {
  // Test case - Two data are splitted into two.
  std::vector<DataImageLabel> dataset;
  const unsigned int numData = 2;
  const unsigned int numK = 2;
  
  for (int i = 0; i < numData; ++i) {
    DataImageLabel data;
    dataset.push_back(data);
  }
  
  std::vector<int> training, validation;
  cross_val_.SplitData(dataset, numK, 0, training, validation);

  unsigned int size_training = training.size();
  unsigned int size_validation = validation.size();

  // Check.
  ASSERT_EQ(size_training, 1) << "Return value from the method is not matched with the expection."; 
  ASSERT_EQ(size_validation, 1) << "Return value from the method is not matched with the expection."; 
}

// Test SplitData function.
TEST_F(CrossValidationTest, data_split_3data) {
  // Test case - Three data are splitted into two.
  // Make first set and second set.
  std::vector<DataImageLabel> dataset;
  const unsigned int numData = 3;
  const unsigned int numK = 2;
  
  for (int i = 0; i < numData; ++i) {
    DataImageLabel data;
    dataset.push_back(data);
  }
  
  std::vector<int> training_0, validation_0;
  std::vector<int> training_1, validation_1;
  
  cross_val_.SplitData(dataset, numK, 0, training_0, validation_0);
  cross_val_.SplitData(dataset, numK, 1, training_1, validation_1);
  
  unsigned int size_training_0 = training_0.size();
  unsigned int size_validation_0 = validation_0.size();
  unsigned int size_training_1 = training_1.size();
  unsigned int size_validation_1 = validation_1.size();

  // Check.
  ASSERT_EQ(size_training_0, 2) << "Return value from the method is not matched with the expection."; 
  ASSERT_EQ(size_validation_0, 1) << "Return value from the method is not matched with the expection.";
  ASSERT_EQ(size_training_1, 1) << "Return value from the method is not matched with the expection."; 
  ASSERT_EQ(size_validation_1, 2) << "Return value from the method is not matched with the expection."; 
}

// Test SplitData function.
TEST_F(CrossValidationTest, data_split_10data_3) {
  // Test case - 10 data is splitted into three.
  // Make second set.
  std::vector<DataImageLabel> dataset;
  const unsigned int numData = 10;
  const unsigned int numK = 3;
  
  for (int i = 0; i < numData; ++i) {
    DataImageLabel data;
    dataset.push_back(data);
  }
  
  std::vector<int> training_0, validation_0;
  std::vector<int> training_1, validation_1;
  
  cross_val_.SplitData(dataset, numK, 0, training_0, validation_0);
  cross_val_.SplitData(dataset, numK, 2, training_1, validation_1);
  
  unsigned int size_training_0 = training_0.size();
  unsigned int size_validation_0 = validation_0.size();
  unsigned int size_training_1 = training_1.size();
  unsigned int size_validation_1 = validation_1.size();

  // Check.
  ASSERT_EQ(size_training_0, 7) << "Return value from the method is not matched with the expection."; 
  ASSERT_EQ(size_validation_0, 3) << "Return value from the method is not matched with the expection."; 
  ASSERT_EQ(validation_0[0], 0) << "Return value from the method is not matched with the expection."; 
  ASSERT_EQ(training_0[0], 3) << "Return value from the method is not matched with the expection."; 
  ASSERT_EQ(size_training_1, 6) << "Return value from the method is not matched with the expection."; 
  ASSERT_EQ(size_validation_1, 4) << "Return value from the method is not matched with the expection."; 
  ASSERT_EQ(validation_1[0], 6) << "Return value from the method is not matched with the expection."; 
  ASSERT_EQ(training_1[0], 0) << "Return value from the method is not matched with the expection."; 
}
  
// Test SplitData function.
TEST_F(CrossValidationTest, data_split_3data_aug) {
  // Test case - Split three data with augmented data.
  std::vector<DataImageLabel> dataset;
  const unsigned int numData = 9;
  const unsigned int numK = 3;
  
  for (int i = 0; i < numData; ++i) {
    DataImageLabel data;
    data.file_name_ = i+1;
    dataset.push_back(data);
  }
  
  const std::string aug = "_aug_";
  
  // Set augmented file name.
  // Test example.
  // [index] [original data] [original/augmented]
  // 0       0               original 1
  // 1       0               augmented 1-1
  // 2       2               original 2
  // 3       2               augmented 2-1
  // 4       2               augmented 2-2
  // 5       5               original 3
  // 6       5               augmented 3-1
  // 7       5               augmented 3-2
  // 8       5               augmented 3-3
  dataset[1].file_name_ = dataset[1].file_name_ + aug;
  dataset[3].file_name_ = dataset[3].file_name_ + aug;
  dataset[4].file_name_ = dataset[4].file_name_ + aug;
  dataset[6].file_name_ = dataset[6].file_name_ + aug;
  dataset[7].file_name_ = dataset[7].file_name_ + aug;
  dataset[8].file_name_ = dataset[8].file_name_ + aug;  
  
  std::vector<int> training_0, validation_0;
  std::vector<int> training_1, validation_1;
  std::vector<int> training_2, validation_2;
  
  // Expected result: 7 training data 1 validation data
  cross_val_.SplitData(dataset, numK, 0, training_0, validation_0);
  // Expected result: 6 training data 1 validation data
  cross_val_.SplitData(dataset, numK, 1, training_1, validation_1);
  // Expected result: 5 training data 1 validation data
  cross_val_.SplitData(dataset, numK, 2, training_2, validation_2);
  
  unsigned int size_training_0 = training_0.size();
  unsigned int size_validation_0 = validation_0.size();
  unsigned int size_training_1 = training_1.size();
  unsigned int size_validation_1 = validation_1.size();
  unsigned int size_training_2 = training_2.size();
  unsigned int size_validation_2 = validation_2.size();

  // Check.
  ASSERT_EQ(size_training_0, 7) << "Return value from the method is not matched with the expection."; 
  ASSERT_EQ(size_validation_0, 1) << "Return value from the method is not matched with the expection.";
  ASSERT_EQ(size_training_1, 6) << "Return value from the method is not matched with the expection."; 
  ASSERT_EQ(size_validation_1, 1) << "Return value from the method is not matched with the expection."; 
  ASSERT_EQ(size_training_2, 5) << "Return value from the method is not matched with the expection."; 
  ASSERT_EQ(size_validation_2, 1) << "Return value from the method is not matched with the expection."; 
}
  
  
} // namespace vehicle_detector
} // namespace robotics
