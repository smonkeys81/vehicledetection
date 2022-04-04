# Copyright 2020 Robotics, Inc. All Rights Reserved.

# This code calculate mean values of the entire images in the specified directory.

import glob
import cv2
import numpy
import argparse
import os

# Parse arguements.
parser = argparse.ArgumentParser(description='Calculate images mean.')
parser.add_argument('--dir', dest='dir', help='Path to image directory', default=None, type=str)
args = parser.parse_args()

n = 0
tot_avg_color = numpy.array([0, 0, 0])

entries = args.dir + '/*.png'
for filename in glob.glob(entries):
    n = n + 1
    myimg = cv2.imread(filename)
    avg_color_per_row = numpy.average(myimg, axis=0)
    avg_color = numpy.average(avg_color_per_row, axis=0)
    tot_avg_color = ( tot_avg_color * (n - 1) + avg_color ) / n
    
    if n % 50 == 0:
      print('%d images were processed: mean ' % n)
      print(tot_avg_color)

print("Final result (BGR)")
print(tot_avg_color)
