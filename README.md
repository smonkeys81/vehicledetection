Copyright © 2019 Robotics, Inc. All Rights Reserved.

# Vehicle Detection Pipeline
This is a vehicle detection pipeline based on [Faster R-CNN](https://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf).
The vehicle detection task is to identify all the vehicles appearing on an input color image.

## Required dependencies

### [Caffe](https://caffe.berkeleyvision.org/)
This pipeline is built based on Caffe library, a deep learning framework.
To use the Faster R-CNN, this repo pulls the Dev version of the Caffe. (Please download from [here](https://github.com/D-X-Y/caffe-faster-rcnn/tree/dev))
Some of the libraries are already included in this repo and other required libraries need to install separately. See the below.

- [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) 6.x+ (tested on 9.0 and 10.0. Note, Ubuntu 16.04 requires CUDA 8+ for compatibility)
- [cuDNN](https://developer.nvidia.com/cudnn) (for GPU acceleration) 6+ (tested on 7.1.2 and 7.3.0)
- BLAS vis ATLAS
- Boost 1.55+ (tested on 1.58)
- Python 2.7 or 3.3+
- protobuf, glog, gflags, gdf5

To install the dependencies above, except for CUDA and cuDNN, please install below by apt-get.

```
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get install --no-install-recommends libboost-all-dev
sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev
```

To build the Caffe, do the followings:

```bash
cd ${Caffe}
mkdir build
cd build/
cmake ..
make -j4 install
```

Troubleshooting: If you see the following message "warning: "THRUST_CUB_NS_PREFIX" redefined", you need to modify the CUDA configuration file.

```bash
sudo gedit /usr/local/cuda/include/thrust/system/cuda/config.h
```

Then, replace below

```
#define THRUST_CUB_NS_PREFIX namespace thrust {   namespace cuda_cub {
#define THRUST_CUB_NS_POSTFIX }  }
```

with 

```
#ifndef THRUST_CUB_NS_PREFIX
#define THRUST_CUB_NS_PREFIX namespace thrust {   namespace cuda_cub {
#endif
#ifndef THRUST_CUB_NS_POSTFIX
#define THRUST_CUB_NS_POSTFIX }  }
#endif
```


### Other dependencies

Most of the dependencies are redundant with those of Caffe, but additional dependencies are:

- OpenCV 3.0+ (2.x and 4.x are not compatiable. Tested on 3.3.1, which is installed with ROS kinetic)
  -* CUDA version for OpenCV and Caffe should be the same. Otherwise, building Caffe using OpenCV result in errors.
- Cmake 2.8.7+ (tested on 3.5.1)


### Latex documents
Install latex to generate documents

```bash
sudo apt-get install texlive-latex-base
sudo apt-get install texlive-latex-extra
sudo apt-get install ko.tex-base
```

## How to prepare dataset

This pipeline is to be tested against the [KITTI dataset](http://www.cvlibs.net/datasets/kitti/). 
To download the publicly available dataset, do the following.

### KITTI dataset

Download KITTI **2D Object** dataset from the 2D object [LINK](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d) of KITTI.

  - [Left color images of object data set (12GB)](http://www.cvlibs.net/download.php?file=data_object_image_2.zip): This data is used for training and test.
  
  - [Right color images of object data set (12GB)](http://www.cvlibs.net/download.php?file=data_object_image_3.zip): This data is used for stereo matching to estimate 3D location during test.

  - [Training labels of object data set (5MB)](http://www.cvlibs.net/download.php?file=data_object_label_2.zip)
  
### Default dataset location

The location of dataset can be moved and configured from *.pt and *.json files.
Below is the default location

 - Training images (left): /media/dataset/kitti/2d/training_img/

 - Training labels: /media/dataset/kitti/2d/training_label/
 
 - Test images (left): /media/dataset/kitti/2d/test_img/
 
 - Test lables: /media/dataset/kitti/2d/test_label/
 
 - Right images: /media/dataset/kitti/2d/data_object_image_3/

## Building Executable
Link the directories

```bash
cd ${VEHICLE_DETECTION_ALPHA}
mkdir build
cd build/
cmake ..
make -j4
```

## Hardware
The Faster R-CNN using the pretrained [VGGNet](https://arxiv.org/pdf/1409.1556.pdf) is the base of this pipeline and requires at least 4GB memory on a Nvidia GPU board.

## Running pipeline

### Do the unit tests

Once the pipeline is built without any issues or errors, you should run the unittests right after.

```bash
cd ${VEHICLE_DETECTION_ALPHA}/build
./test/unit_tests
```

### Training a model

To train a model, you can either run the following script or the executable

```bash
cd ${VEHICLE_DETECTION_ALPHA}/build/scripts
./train.sh
```

```bash
cd ${VEHICLE_DETECTION_ALPHA}/build/tools
./train_model ../config/config.json ../model/VGG16_faster_rcnn_solver.pt ../model/VGG16.v2.caffemodel
```

### Conversion of the trained model into one for the testing.

Run the following scripts or the excutable to convert the weight file (*.caffemodel) into the one for the testing.

```bash
cd ${VEHICLE_DETECTION_ALPHA}/build/scripts
./convert_model.sh [Iteration_number]
```

```bash
cd ${VEHICLE_DETECTION_ALPHA}/build/scripts
python convert_model.py --model [test_network.pt] --weights [trained_weights.caffemodel] \
    --config [default_config.json] --net_out [output_filename.caffemodel]
```

### Test an image

### Performance Evaluation

To be updated

## C++ component

To be updated
