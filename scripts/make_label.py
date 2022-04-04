# Copyright 2020 Robotics, Inc. All Rights Reserved.

# This code generates BDD100K label file for the specified image directory.

from os import listdir
from os.path import isfile, join
import argparse
import json

# Parse arguements.
parser = argparse.ArgumentParser(description='My test')
parser.add_argument('--file', dest='file_name', help='Path to file', default=None, type=str)
parser.add_argument('--dir', dest='dir_name', help='Path to image directory', default=None, type=str)
args = parser.parse_args()

# Find all files in directory.
onlyfiles = [f for f in listdir(args.dir_name) if isfile(join(args.dir_name, f))]
print(onlyfiles)

with open('/media/dataset/bdd100k/labels/bdd100k_labels_images_val.json') as f:
  data = json.load(f)

print(data)
