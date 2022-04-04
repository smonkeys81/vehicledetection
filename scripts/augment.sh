# This script is used to augment dataset.

# arguments:
# 1. Configuration file.

../tools/augmentor  ../config/config.json \

# Convert labels into VOC format.
./convert_label.sh kitti

mv ../model/out/kitti.trainval ../model/out/kitti_aug.trainval
