# This script evaluates performance of vehicle detection module based on faster RCNN.

# arguments:
# 1. Name of Dataset.
# 2. Configuration file.
# 3. Network model name.
# 4. Weight file.

../tools/evaluate kitti \
                  ../config/config.json \
                  VGG16 \
                  ../model/out/VGG16_faster_rcnn_converted.caffemodel
