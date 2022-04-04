// Copyright © 2019 Robotics, Inc. All Rights Reserved.

// This file provides methods for cross-validation for vehicle detection.

#include "cross_validation.h"


namespace robotics {
namespace vehicle_detector {
  
  
// Constructor.
CrossValidation::CrossValidation() {
}
  
// Execute cross validation procedure.
bool CrossValidation::CrossValidationDatasetKITTI(const std::string& val_config_file,
                                                  const std::string& model_name) {
  // Load parameters.
  if(!LoadParamCrossVal(val_config_file)) {
    ErrMsg(__func__, "Parameter loading failed:", val_config_file);
    return false;
  }
  std::cout << "Parameter loading completed." << std::endl;
  
  // In the cross validation parameters, num_k should be greater than 1.
  if(num_k_ < 2) {
    std::cerr << __func__ << " k(=" << num_k_ << ") should be greater than 1." << std::endl;
    return false;
  }
  
  // Load dataset.
  if(!kitti_.LoadDataSet(dir_label_, dir_image_, dir_image_right_)) {
    ErrMsg(__func__, "Unable to load dataset.");
    return false;
  }
  std::cout << "Dataset loading completed." << std::endl;
  
  // Sort dataset by filename.
  sort(kitti_.dataset_.begin(), kitti_.dataset_.end(),
       [](DataImageLabel const& v1, DataImageLabel const& v2) {
         return v1.file_name_ < v2.file_name_;
       } );
  
  std::vector<std::string> path_label_orig, path_img_orig;
  
  // validate k times.
  for(int i = 0; i < num_k_; ++i) {
    std::cout << "************************" << std::endl;
    std::cout << "*                      *" << std::endl;
    std::cout << "* Validation step: " << i+1 << "/" << num_k_ << " *" << std::endl;
    std::cout << "*                      *" << std::endl;
    std::cout << "************************" << std::endl;
   
    // Make training and validation index list (Set dividing).
    std::vector<int> idx_training, idx_validation;
    if(!SplitData(kitti_.dataset_, num_k_, i, idx_training, idx_validation)) {
      ErrMsg(__func__, "Failed to split data.");
      return false;
    }
    std::cout << "Total data: " << kitti_.dataset_.size() 
      << " Training data: " << idx_training.size()
      << " (" << idx_training[0] << "~" << idx_training[idx_training.size()-1]
      << ")"
      << " Validation data: " << idx_validation.size()
      << " (" << idx_validation[0] << "~" << idx_validation[idx_validation.size()-1]
      << ")" << std::endl;
    
    // Convert training label files into VOC format.
    std::cout << "************************" << std::endl;
    std::cout << "* Convert label files  *" << std::endl;
    std::cout << "************************" << std::endl;

    // Open output file.
    std::ofstream out_file;
    std::stringstream stream_voc;
    stream_voc << "Cross_val_" << i << "_" << num_k_-1 << ".trainval";
    out_file.open(stream_voc.str());
    if(!out_file) {
      ErrMsg(__func__, "Unable to open out file:", stream_voc.str());
      return false;
    }
    unsigned int seq = 0;
    for(int j = 0; j < idx_training.size(); ++j) {
      // Get file name from the divided training indices.
      const unsigned int index = idx_training[j];
      std::string img_file_name = kitti_.dataset_[index].file_name_;
      std::string label_file_name = kitti_.dataset_[index].file_name_;
      label_file_name.replace(label_file_name.size()-3, 3, "txt");
      
      // Write labels in the output file.
      kitti_.SaveVOC(out_file, kitti_.dataset_[index].gt_vehicles_, seq, img_file_name, kTopLeftWidthHeight);
    }
    out_file.close();
    
    // For each set, train a model.
    std::cout << "************************" << std::endl;
    std::cout << "*    Model training    *" << std::endl;
    std::cout << "************************" << std::endl;

    PerformanceEval eval(config_file_);
    
    // Get index of backbone.
    unsigned int index_model;
    if(!eval.param_.GetBackboneNum(model_name, index_model)) {
      ErrMsg(__func__, "Invalid backbone name", model_name);
      return false;
    }
    
    // Preparation - Modify train.pt file
    // Change training labels.
    std::string line_origin;
    const std::string line_source = "    source: \"" + stream_voc.str() + "\"";
    UpdateTrainInfo(eval.param_.file_solver_[index_model], "source",
                    line_source, line_origin);
    path_label_orig.push_back(line_origin);
    // Change image directory.
    const std::string line_root_folder = "    root_folder: \"" + dir_image_ + "\"";
    UpdateTrainInfo(eval.param_.file_solver_[index_model], "root_folder",
                    line_root_folder, line_origin);
    path_img_orig.push_back(line_origin);
    
    // Train.
    if(!eval.TrainModelByName(model_name)) {
      ErrMsg(__func__, "Training failed.");
      return false;
    }

    // Make result directory for each fold.
    std::string new_dir = "cross_val_" + std::to_string(i+1) + "_" + std::to_string(num_k_);
    mkdir(new_dir.c_str(), 0777);

    // Plot loss and move files.
    std::string bash = "python plot_loss.py --file loss_*";
    system(bash.c_str());
    bash = "mv loss_po* " + new_dir;
    system(bash.c_str());
    // Move training label file.
    bash = "mv *.trainval " + new_dir;
    system(bash.c_str());

    unsigned int iteration = num_iter_from_;

    // Check right image for 3D estimation.
    eval.kitti_.exist_img_right_ = kitti_.exist_img_right_;

    while (iteration <= num_iter_to_) {
      // Convert the trained model by executing a bash script.
      std::cout << "************************" << std::endl;
      std::cout << "*   Model converting   *" << std::endl;
      std::cout << "************************" << std::endl;

      bash = file_conv_bash_ + " " + std::to_string(iteration);
      system(bash.c_str());
  
      // Copy test set into the evaluation instance.
      eval.kitti_.dataset_.clear();
      for(int j = 0; j < idx_validation.size(); ++j) {
        int index = idx_validation[j];
        eval.kitti_.dataset_.push_back(kitti_.dataset_[index]);
      }
      
      // Evaluate the converted model - prepare model and parameters.
      std::cout << "************************" << std::endl;
      std::cout << "*      Evaluation      *" << std::endl;
      std::cout << "************************" << std::endl;

      // Error message is printed in the function.
      if(!eval.LoadAll(model_name, trained_file_)) {
        return false;
      }
    
      // Evaluate the converted model.
      if(!eval.EvaluateDataset(kKITTI, false)) {
        ErrMsg(__func__, "Error occurred during evaluation.");
        return false;
      }   
  
      // Visualize result.
      if(eval.data_all_.size() > 0) {
        int idx_iou, idx_thres;
        eval.VisualizePlotLineResult();
        eval.VisualizePlotRadResult(idx_iou, idx_thres);
      }  

      // Make sub result directory.
      std::string new_dir_sub = new_dir + PathSeparator() + "iter" + std::to_string(iteration);
      mkdir(new_dir_sub.c_str(), 0777);      
      
      bash = "mv birdeye " + new_dir_sub;
      system(bash.c_str());
      bash = "mv Result_* " + new_dir_sub;
      system(bash.c_str());
      bash = "mv ../model/out/VGG16_faster_rcnn_converted* " + new_dir_sub;
      system(bash.c_str());

      iteration += num_iter_step_;
    }
    
    // Restore train.pt file.
    std::string dummy_str;
    UpdateTrainInfo(eval.param_.file_solver_[index_model], "source", path_label_orig[0], dummy_str);
    UpdateTrainInfo(eval.param_.file_solver_[index_model], "root_folder", path_img_orig[0], dummy_str);
    
  } 

  return true;
}
  
/// Split dataset into training set and validation set.
bool CrossValidation::SplitData(const std::vector<DataImageLabel> dataset,
                                const unsigned int num_k, const unsigned int trial,
                                std::vector<int> &idx_training,
                                std::vector<int> &idx_validation) {
  // Exception.
  if(num_k < 1 || trial >= num_k) {
    std::cerr << "Invalid parameter." << std::endl;
    return false;
  } 
  
  // Count the number of augmented files.
  const unsigned int size_all = dataset.size();
  unsigned int size_orig = size_all;
  for(int i = 0; i < size_all; ++i) { 
    if (dataset[i].file_name_.find(aug) != std::string::npos) {
      size_orig--;
    }
  }

  // The indices between start and end, from the original set, become validation set.
  const unsigned int start = int(size_orig * (trial / (float)num_k));
  const unsigned int end = int(size_orig * ((trial + 1) / (float)num_k) - 1);
  // Case 1: The number of data divided by k(num_k) is a natural number.
  // - The number of validation data is always the same.
  // - Example: 12 data, 3 folds.
  // -- Trial 0: validation 0 1 2 3, training 4 5 6 7 8 9 10 11
  // -- Trial 1: validation 4 5 6 7, training 0 1 2 3 8 9 10 11
  // -- Trial 2: validation 8 9 10 11, training 0 1 2 3 4 5 6 7
    
  // Case 2: The number of data divided by k(num_k) is NOT a natural number.
  // - The validation data count rounds off decimal point.
  // - Example: 10 data, 3 folds.
  // -- Trial 0: validation(3) 0 1 2, training(7) 3 4 5 6 7 8 9
  // -- Trial 1: validation(3) 3 4 5, training(7) 0 1 2 6 7 8 9
  // -- Trial 2: validation(4) 6 7 8 9, training(6) 0 1 2 3 4 5 
  // - Example: 11 data, 3 folds.
  // -- Trial 0: validation(3) 0 1 2, training(8) 3 4 5 6 7 8 9 10
  // -- Trial 1: validation(4) 3 4 5 6, training(7) 0 1 2 7 8 9 10
  // -- Trial 2: validation(4) 7 8 9 10, training(7) 0 1 2 3 4 5 6 
  // - Example: 10 data, 4 folds.
  // -- Trial 0: validation(2) 0 1, training(8) 2 3 4 5 6 7 8 9
  // -- Trial 1: validation(3) 2 3 4, training(7) 0 1 5 6 7 8 9
  // -- Trial 2: validation(2) 5 6, training(8) 0 1 2 3 4 7 8 9
  // -- Trial 3: validation(3) 7 8 9, training(7) 0 1 2 3 4 5 6

  unsigned int index_orig = -1;
  for(int i = 0; i < size_all; ++i) {

    // Increase original data index if the file name doesn't have "aug" in it.
    if (dataset[i].file_name_.find(aug) == std::string::npos) {
      index_orig++;
    }
    
    // Validation set does not include augmented data.
    // Both boundary is also included.
    if(index_orig >= start && index_orig <= end) {
      if (dataset[i].file_name_.find(aug) == std::string::npos) {
        idx_validation.push_back(i);
        
      }      
    // Traning set has both original and augmented data.
    } else {
      idx_training.push_back(i);
    }
  }
  
  return true;
}

// Find and update training configuration file with the desired training labels and directory.
bool CrossValidation::UpdateTrainInfo(const std::string& solver_file,
                                      const std::string& pattern,
                                      const std::string& line_input,
                                      std::string& line_original) {
  // Load solver parameter.
  caffe::SolverParameter solver_param;
  caffe::ReadSolverParamsFromTextFileOrDie(solver_file, &solver_param);
  
  // Input train file.
  std::ifstream input_file(solver_param.train_net());
  std::vector<std::string> lines;
  std::string input;
  while (std::getline(input_file, input)) {
      lines.push_back(input);
  }
  input_file.close();
  
  // Open train file as output.
  std::ofstream output_file(solver_param.train_net());
  for (auto& line : lines) {
    if (line.find(pattern) != std::string::npos) {
      output_file << line_input << std::endl;
      // backup original path.
      line_original = line;
    } else {
      output_file << line << std::endl;
    }
  }

  return true;
}
  
  
} // namespace vehicle_detector
} // namespace robotics
