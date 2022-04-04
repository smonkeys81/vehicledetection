# Copyright 2020 Robotics, Inc. All Rights Reserved.

# This code calculates average box of the vehicles in specified range in dataset.

from matplotlib import pyplot as plt
import sys
import glob
import argparse
import os
import math
import numpy as np

# Parse arguements.
parser = argparse.ArgumentParser(description='Calculate average box area')
parser.add_argument('--label', dest='label', help='Directory to KITTI label file', default=None, type=str)
parser.add_argument('--int', dest='interval', help='Interval in meters', default=15, type=int)
parser.add_argument('--far', dest='far', help='Far distance in meters', default=200, type=int)
args = parser.parse_args()
interv = args.interval

# Define indices for KITTI.
idx_left = 4
idx_top = 5
idx_right = 6
idx_bottom = 7
idx_3dx = 11
idx_3dy = 12
idx_3dz = 13

# Original anchor boxes from ImageNet - replace with your own.
anchor = [-84, -40, 100, 56,
             -176,  -88,  192,   104,
             -360,  -184,   376,   200,
             -56,   -56,    72,    72,
             -120,  -120,   136,   136,
             -248,  -248,   264,   264,
             -36,   -80,    52,    96,
             -80,   -168,   96,    184,
             -168,  -344,   184,   360]
anchor_default = np.reshape(anchor, (-1, 4))


# Load all labels.
data = []
entries = args.label + '/*.txt'
for filename in glob.glob(entries):
  # Open label file.
  with open(filename, mode='rt') as f:
    for line in f:
      split = line.split(' ')
      # Find car, truck and van.
      if split[0] == 'Car' or split[0] == 'Truck' or split[0] == 'Van':
        # Get box width and height.
        width = round(float(split[idx_right]) - float(split[idx_left]), 2)
        height = round(float(split[idx_bottom]) - float(split[idx_top]), 2)
        # Calc distance from ego-vehicle.
        x = float(split[idx_3dx])
        z = float(split[idx_3dz])
        dist = math.sqrt(x*x + z*z)
        
        data.append([width, height, width*height, x, z, dist])

# Make numpy array and sort.
np_data = np.array(data)
sorted_data = np_data[np.argsort(np_data[:,5])]


# Format plots
size_fig_w = 550
size_fig_h = 450
fig = plt.figure(figsize=(12,12))   
plt.xlim(0, size_fig_w)
plt.ylim(0, size_fig_h)
size_dot = 35

# plot all samples.
plt.scatter(sorted_data[:,0], sorted_data[:,1], s=size_dot, c='gray', edgecolors='none', alpha=0.15)

# 1st trial by default anchors.
plt.scatter(anchor_default[:,2]-anchor_default[:,0], anchor_default[:,3]-anchor_default[:,1], s=size_dot, c='blue', edgecolors='g', alpha=1, marker = 's', label='Default Anchors')


dist_min = 0
dist_max = interv
idx_start = 0
idx_end = 0
stats = []
print("Total occurrence: %d" % sorted_data.shape[0])

for idx in range(sorted_data.shape[0]):
  dist = sorted_data[idx][5]

  if(dist >= dist_min and dist < dist_max and idx < sorted_data.shape[0]):
    idx_end = idx

    if (idx == sorted_data.shape[0]-1):
      mean_w = np.mean(sorted_data[idx_start:idx_end, 0])
      std_w = np.std(sorted_data[idx_start:idx_end, 0])
      mean_h = np.mean(sorted_data[idx_start:idx_end, 1])
      std_h = np.std(sorted_data[idx_start:idx_end, 1])
      mean_dist = np.mean(sorted_data[idx_start:idx_end, 5])
      std_dist = np.std(sorted_data[idx_start:idx_end, 5])

      stats.append([idx_start, idx_end, mean_w, std_w, mean_h, std_h, mean_dist, std_dist])
  else:
    # calulate means.
    mean_w = np.mean(sorted_data[idx_start:idx_end, 0])
    std_w = np.std(sorted_data[idx_start:idx_end, 0])
    mean_h = np.mean(sorted_data[idx_start:idx_end, 1])
    std_h = np.std(sorted_data[idx_start:idx_end, 1])
    mean_dist = np.mean(sorted_data[idx_start:idx_end, 5])
    std_dist = np.std(sorted_data[idx_start:idx_end, 5])

    stats.append([idx_start, idx_end, mean_w, std_w, mean_h, std_h, mean_dist, std_dist])
      
    dist_min += interv
    dist_max += interv
    idx_start = idx
    idx_end = idx
       
print ("<Statistics>")

anchor_mean = []
anchor_stddev1 = []
anchor_stddev2 = []

range = 0
n = 0
for i in stats:
  anchor_mean.append([i[2], i[4]])
  if i[6] <= args.far:
    anchor_stddev1.append([i[2]+i[3], i[4]+i[5]])
    anchor_stddev1.append([i[2]-i[3], i[4]-i[5]])
  anchor_stddev2.append([i[2]+i[3], i[4]-i[5]])
  anchor_stddev2.append([i[2]-i[3], i[4]+i[5]])
  print ("Zone [%02d-%02dm]" %(range, range+interv))
  print ("Occurrence Mean_w Stddev_w Mean_h Stddev_h Mean_dist Stddev_dist")
  print ("%d %0.3f %0.3f %0.3f %0.3f %0.3f %0.3f" %(i[1]-i[0]+1, i[2], i[3], i[4], i[5], i[6], i[7]))
  range = range + interv
print("\n")

np_mean = np.array(anchor_mean)
np_mean_std1 = np.array(anchor_stddev1)
np_mean_std2 = np.array(anchor_stddev2)

plt.scatter(np_mean_std1[:,0], np_mean_std1[:,1], s=size_dot, c='y', edgecolors='none', alpha=1, label='Mean+-sigma(wh)')
plt.scatter(np_mean_std2[:,0], np_mean_std2[:,1], s=size_dot, c='g', edgecolors='none', alpha=1, label='Mean+-sigma(wh)')
plt.scatter(np_mean[:,0], np_mean[:,1], s=size_dot, c='r', edgecolors='none', alpha=1, label='Mean')
plt.legend()

proposed_box = np.concatenate((np_mean, np_mean_std1, np_mean_std2), axis=0)

print ("<Proposed %d anchor box configuration>" % (np_mean.shape[0] + np_mean_std1.shape[0]+ np_mean_std2.shape[0]) )
for i in proposed_box:
  print ("%d, %d, %d, %d," % (round(-i[0]*0.5), round(-i[1]*0.5), round(i[0]*0.5), round(i[1]*0.5)))

# Generate file name.
fig_file = "recommended_box.png"

# Sava image.
plt.savefig(fig_file, bbox_inches='tight')
