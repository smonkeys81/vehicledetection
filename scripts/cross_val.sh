# This script evaluates performance of vehicle detection module based on faster RCNN.

# arguments:
# 1. Configuration file.
# 2. Network model name.

../tools/cross_val  config_crossval.json \
                    VGG16
