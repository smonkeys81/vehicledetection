# Copyright 2020 Robotics, Inc. All Rights Reserved.

# This code merges two VOC label files to combine two datasets.

import argparse

# Parse arguements.
parser = argparse.ArgumentParser(description='Merge training labels.')
parser.add_argument('--file1', dest='f1', help='First label file', default=None, type=str)
parser.add_argument('--file2', dest='f2', help='Second label file', default=None, type=str)
parser.add_argument('--dest', dest='dest', help='Destination file name', default=None, type=str)
args = parser.parse_args()

# Open files.
r1 = open(args.f1, mode='rt')
r2 = open(args.f2, mode='rt')
w = open(args.dest, mode='wt')

count = 0
for line in r1:
  w.writelines(line)
  if line[0] == '#':
    count=count+1

print("First label file contains %d image files" % count)
r1.close()

count_2 = 0
for line in r2:
  if line[0] == '#':
    data = "# %d\n" % count
    w.writelines(data)
    count=count+1
    count_2=count_2+1
  else:
    w.writelines(line)

print("Second label file contains %d image files" % count_2)
print("Merged label file contains %d image files" % count)

w.close()

