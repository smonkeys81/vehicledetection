# This script merges two VOC label files for training.

# arguments:
# 1. First label file
# 2. Second label file
# 3. Destination file

#!/usr/bin/env sh

python merge_labels.py \
    --file1 ../model/out/bdd.trainval \
    --file2 ../model/out/kitti.trainval \
    --dest ../model/out/bdd_kitti.trainval
