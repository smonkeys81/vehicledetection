// Copyright © 2019 Robotics, Inc. All Rights Reserved.

// This file provides basic utility functions.


#ifndef ROBOTICS_VEHICLEDETECTION_UTIL_H_
#define ROBOTICS_VEHICLEDETECTION_UTIL_H_


#include <iostream>
#include <fstream>
#include <string>
#include <dirent.h>
#include <json.h>
#include <random>
#include <chrono>
#include <sys/stat.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace robotics {
namespace vehicle_detector {


/// \brief Make all input string upper case.
/// \param[in] str Input string.
/// \return Uppercase string.
inline std::string MakeUpperCase(std::string str) {
  std::transform(str.begin(), str.end(), str.begin(), ::toupper);
  return str;
}
  
/// \brief Print error messages and the name of function resulted in the error.
/// \param[in] function_name Name of function.
/// \param[in] msg Error message to display.
/// \param[in] param Additional parameter information to display.
inline void ErrMsg(std::string function_name, std::string msg, std::string param = "") {
  std::cerr << function_name << "() - " << msg << " " << param << std::endl;
}

/// \brief Provide path separator.
/// \return Path separator.
inline std::string PathSeparator() {
  return "/";
}

/// \brief Calculate radian from degree.
/// \param[in] deg Amoung of rotation.
/// \return deg in radian.
template <class T>
inline T DegToRad(T deg) {
  return deg * CV_PI / 180.;
}

/// \brief Calculate degree from radian.
/// \param[in] rad Amoung of rotation.
/// \return degree value.
template <class T>
inline T RadToDeg(T rad) {
  return rad * 180. / CV_PI;
}
  
/// \brief Write text at the given position (on box) in input image.
/// \param[in] image Image to write.
/// \param[in] pos_x Position x.
/// \param[in] pos_y Position y.
/// \param[in] text Text to write.
/// \param[in] font_scale Size of the text.
/// \param[in] thickness Thickness of the text.
inline void WriteText(cv::Mat& image,
                      const unsigned int pos_x,
                      const unsigned int pos_y,
                      const std::string& text,
                      const double font_scale = 1.0,
                      const double thickness = 1.0) {
  int baseline = 0;
  
  // Get text size.
  cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_PLAIN,
                                      font_scale, thickness, &baseline);
  
  int pos_y_new = pos_y - textSize.height;
  if(pos_y_new < 0) {
    pos_y_new = 0;
  }
  
  // Draw background.
  cv::rectangle(image, cv::Point(pos_x, pos_y_new),
                cv::Point(pos_x + textSize.width, pos_y_new + textSize.height),
                cv::Scalar::all(180), CV_FILLED);
  // Write text.
  cv::putText(image, text, cv::Point(pos_x, pos_y_new + textSize.height),
              cv::FONT_HERSHEY_PLAIN, font_scale, cv::Scalar::all(0), thickness, 8);
}

/// \brief This function checks if input value is out of boundary (min&max), and returns boundary if the valus is out of range.
/// \param[in] input Input value to be checked.
/// \param[in] min Minimum boundary.
/// \param[in] max Maximum boundary.
/// \return Either original input value or boundary value.
inline float SetBoundary(const float input, const float min, const float max) {
  float output;
  output = std::max(input, min);
  output = std::min(output, max);
  return output;
}
  
/// \brief Generate 1 with probability p and 0 with probability 1-p, where 0 <= p <=1.
/// \param[in] prob Probability.
/// \return 1 or 0 when the passing parameter is correct, -1 otherwise.
inline int GetRandomZeroOne(const float prob = 0.5) {
  // Check exception.
  if(prob > 1 || prob < 0) {
    std::cerr << "Parameter is out of range." << std::endl;
    return -1;    
  }
  
  // This generates 1 with probability prob (p).
  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> distrib({1-prob, prob});
  
  return distrib(gen);
}
  
/// \brief Generate random number between two values.
/// \param[in] min Minimum random value.
/// \param[in] max Maximum random value.
/// \return Random number between min and max.
inline float GetRandomReal(const float min, const float max) {
  // Check exception.
  if(min >= max) {
    std::cerr << "Parameter has wrong range." << std::endl;
    return -1;    
  }
  
  // This generates uniform distribution .
  // P(x|min, max) = 1 / (max - min), min <= x < max.
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> distrib(min, max);
  
  return distrib(gen);
}
  
/// \brief Method loading a model.
/// \return Path of the output directory for unit tests.
inline std::string GetTestOutputDir() {
  return PACKAGE_TEST_PATH; // given by CMakeLists.txt
}

/// \brief Method to check if file exists.
/// \param[in] name Name of the file including path.
/// \return True if the file exists, false otherwise.
inline bool FileExist (const std::string& name) {
    std::ifstream f(name.c_str());
  return f.good();
}

/// \brief Method to check if file exists.
/// \param[in] path Directory path to be opened.
/// \param[out] dir Pointer to directory.
/// \param[in] function_name Name of the function who called this function.
/// \return True if opening directory was successful.
inline bool OpenDirectory(const std::string& path, DIR** dir,
                          const std::string& function_name = "") {
  std::cout << function_name << "() - Open directory: " << path << std::endl;
  if ((*dir = opendir(path.c_str())) == nullptr) {
    std::cerr << function_name << "(): Path (" << path << ") doesn't exist." << std::endl;
    return false;
  }
  
  return true;
}  

/// \brief Method to open Json file.
/// \param[in] file Path to Json file.
/// \param[out] root Json values.
/// \return True if the input file exists and is opened successfully.
inline bool OpenFileJSON(const std::string& file, Json::Value &root) {
  // Open file.
  std::ifstream inFile;
  inFile.open(file);
  if(!inFile) {
    std::cerr << __FUNCTION__ << ": Unable to open file: " 
        << file << std::endl;
    return false;
  }
  
  //Json::Value root;  
  Json::CharReaderBuilder builder;
  builder["collectComments"] = true;
  JSONCPP_STRING errs;
  if (!parseFromStream(builder, inFile, &root, &errs)) {
    std::cerr << errs << std::endl;
    inFile.close();
    return false;
  }
  return true;
}
  
/// \class PredictionInfo
/// This provides score comparing method between scores for sorting of predictions.
class PredictionInfo {
public:
  /// \brief Score of the predicted bbox.
  float score_;
  
  /// \brief Pointer to each of prediction.
  const float* head_;

  /// \brief Compare score value.
  /// \param[in] pred_info Prediction data.
  /// \return True if input score value is greater than the member score, false otherwise.
  bool operator <(const PredictionInfo& pred_info) {
    return (pred_info.score_ < score_);
  }
};


} // namespace vehicle_detector
} // namespace robotics


#endif // ROBOTICS_VEHICLEDETECTION_UTIL_H_
