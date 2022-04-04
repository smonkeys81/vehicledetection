# This script converts the labels of KITTI/BDD dateset into one label file of VOC format.

# arguments:
# 1. Name of Dataset.
# 2. Path to labels of test dataset.
# 3. Path and file name to write.

#!/usr/bin/env sh

if [ $1 == "kitti" ]; then
../tools/convert_to_voc $1 \
                        /media/dataset/kitti/2d/training_label/ \
                        ../model/out/kitti.trainval
elif [ $1 == "bdd" ]; then
../tools/convert_to_voc $1 \
                        /media/dataset/bdd100k/labels/bdd100k_labels_images_val.json \
                        ../model/out/bdd.trainval
fi
