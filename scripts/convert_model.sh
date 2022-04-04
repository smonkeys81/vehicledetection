# This script converts the trained model into test model.

# arguments:
# $1 Number of iterations to specify trained weights file.
# $2 Number of GPU to use. (default: 0)
# 1. Model file consists of network configurations.
# 2. Trained weights file.
# 3. Configuration file including parameter settings.
# 4. Target file name.

#!/usr/bin/env sh

if [ -z "$2" ]; then
  gpu_no=0
else
  gpu_no=$2
fi

model=VGG16

python convert_model.py \
    --model ../model/${model}/${model}_faster_rcnn_test.pt \
    --weights ../model/out/${model}/${model}_faster_rcnn_iter_$1.caffemodel \
    --config ../model/default_config.json \
    --net_out ../model/out/${model}_faster_rcnn_converted.caffemodel \
    --gpu $gpu_no
