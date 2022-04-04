# This script executes calc_avgbox_kitti.py.

# arguments:
# $1 Distance interval.
# $2 Min avg. distance in a zone to remove additional anchor box.
# 1. Directory where label fils are stored.


#!/usr/bin/env sh

python calc_avgbox_kitti.py \
      --label /media/dataset/kitti/2d/training_label \
      --int $1 --far $2

