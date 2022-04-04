// Copyright © 2019 Robotics, Inc. All Rights Reserved.

// This file is the entry point of the unit test.


#include "gtest/gtest.h"

/// \brief Entry function for unit test. 
/// \param[in] argc Number of arguments.
/// \param[in] argv Character array of arguments.
/// \return True if all procedures were successful.
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TEST();
}
