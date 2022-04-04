# Copyright 2020 Robotics, Inc. All Rights Reserved.

# This code plots width and height of boxes from TP and FN.

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import sys
import glob
import argparse
import os
import math
import numpy as np
import cv2


# Parse arguements.
parser = argparse.ArgumentParser(description=' ')
parser.add_argument('--tp', dest='tp', help='Directory to TP image files', default=None, type=str)
parser.add_argument('--fn', dest='fn', help='Directory to FN image files', default=None, type=str)
args = parser.parse_args()

# Definitions
markers = ['D', 'o', 's', '>', '^', '+', 'x', 'p', 'h', '<', 'D', 'o']
colors = ['red', 'y', 'g', 'royalblue', 'orangered', 'purple', 'rosybrown', 'orchid', 'olive', 'cyan', 'red', 'y', 'g']
lebels = ['FN (Lv0:visible)', 'FN (Lv1:partly occ)', 'FN (Lv2:largely occ)']
xlabel = 'width'
ylabel = 'height'
zlabel = 'distance(m)'
# Maximum distance.
max_dist = 110
max_length = 1920

# Original anchor boxes from ImageNet - replace with your own.
anchor = [-84, -40, 100, 56,
             -176,  -88,  192,   104,
             -200,  -170,   200,   170,
             -56,   -56,    72,    72,
             -120,  -120,   136,   136,
             -150,  -55,   150,   55,
             -36,   -80,    52,    96,
             -80,   -168,   96,    184,
             -150,  -170,   150,   170,
-51,-34,51,34,
-11,-10,11,10,
-28,-21,28,21]
anchor_default = np.reshape(anchor, (-1, 4))


# Load TP and FN images.
data_tp = []
data_fn = []
if args.tp != None:
  entries = args.tp + '/*.jpg'
  for filename in glob.glob(entries):
    # Open image file.
    img = cv2.imread(filename, 1)
    # Store size.
    height, width, ch = img.shape
    # Parsing from filename - difficulty and distance.
    split_result = filename.split("_")
    data_tp.append([float(width), float(height), float(split_result[2]), float(split_result[3])])
if args.fn != None:
  entries = args.fn + '/*.jpg'
  for filename in glob.glob(entries):
    # Open image file.
    img = cv2.imread(filename, 1)
    # Store size.
    height, width, ch = img.shape
    # Parsing from filename - difficulty and distance.
    split_result = filename.split("_")
    data_fn.append([float(width), float(height), float(split_result[2]), float(split_result[3])])
  
np_data_tp = np.array(data_tp)  
np_data_fn = np.array(data_fn)  

# Format plots
size_fig_w = 550
size_fig_h = 450
fig = plt.figure(figsize=(12,12)) 
fig_3d = plt.figure(figsize=(12,12)) 
ax = fig.add_subplot(111)  
ax_3d = fig_3d.add_subplot(111, projection='3d')  
ax.set_xlim(0, size_fig_w)
ax.set_ylim(0, size_fig_h)
ax_3d.set_xlim(0, size_fig_w)
ax_3d.set_ylim(0, size_fig_h)
ax_3d.set_zlim(0, max_dist)

size_dot = 35
size_dot_3d = 25


def draw(x_lim, y_lim, mode):
  xlimit = max_length
  ylimit = max_length
  if x_lim != 0:
    xlimit = x_lim
  if y_lim != 0:
    ylimit = y_lim

  if mode == 'level':
    # plot TP samples.
    if np_data_tp.shape[0] > 0:
      a = 0.2
      mask = (np_data_tp[:,0] <= xlimit) & (np_data_tp[:,1] <= ylimit)
      ax.scatter(np_data_tp[mask, 0], np_data_tp[mask, 1], s=int(size_dot/1.5), c='blue', edgecolors='none', alpha=a,
      label='TP')
  
    # plot FN samples.
    alpha_fn = 0.6
    if np_data_fn.shape[0] > 0:  
      mask = (np_data_fn[:,0] <= xlimit) & (np_data_fn[:,1] <= ylimit)
      for i in range(3):
        mask_temp = (np_data_fn[:,2] == i) & mask
        ax.scatter(np_data_fn[mask_temp, 0], np_data_fn[mask_temp, 1], s=size_dot, c=colors[i], edgecolors='grey', alpha=alpha_fn, marker=markers[i], label=lebels[i])
        ax_3d.scatter(np_data_fn[mask_temp, 0], np_data_fn[mask_temp, 1], np_data_fn[mask_temp, 3], s=size_dot_3d, c=colors[i], edgecolors='grey', alpha=alpha_fn, marker =markers[i], label=lebels[i])

  elif mode == 'dist':
    # plot FN samples.
    alpha_fn = 0.8
    if np_data_fn.shape[0] > 0: 
      mask = (np_data_fn[:,0] <= xlimit) & (np_data_fn[:,1] <= ylimit)
      max_height = int(np.max(np_data_fn[:,3])/10)
      for i in range(max_height+1): 
        mask_temp = (np_data_fn[:,3] >= i*10) & (np_data_fn[:,3] < (i+1)*10) & mask
        if np_data_fn[mask_temp, 0].shape[0] > 0:
          ax.scatter(np_data_fn[mask_temp, 0], np_data_fn[mask_temp, 1], s=size_dot, c=colors[i], edgecolors='grey', alpha=alpha_fn, marker=markers[i], label='Dist: %d-%dm' % (i*10, (i+1)*10))
  
  # Anchors - common.
  mask = (anchor_default[:,0] <= xlimit) & (anchor_default[:,1] <= ylimit)
  ax.scatter(anchor_default[mask,2] - anchor_default[mask,0], anchor_default[mask,3] - anchor_default[:,1], s=size_dot*4, c='w', edgecolors='black', marker = '*', label='Anchor')

  for i in anchor_default:
    x = [i[2]-i[0],i[2]-i[0]] 
    y = [i[3]-i[1],i[3]-i[1]] 
    z = np.array([0, 110])
    ax_3d.plot(x, y, z, c='grey', marker = '*', linewidth=2, markersize=size_dot/3, markerfacecolor='white', label='Anchor')


draw(0, 0, 'level')
title = '[Result] Anchors: %d, TP: %d, FN: %d' % (anchor_default.shape[0], np_data_tp.shape[0],np_data_fn.shape[0])
ax.set_title(title)
ax_3d.set_title(title)
ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
ax_3d.set_xlabel(xlabel)
ax_3d.set_ylabel(ylabel)
ax_3d.set_zlabel(zlabel)
ax_3d.view_init(10,-80)
ax.legend()

# Generate file name.
fig_file = "result_box.png"
fig_file_3d_1 = "result_box_3d_1.png"
fig_file_3d_2 = "result_box_3d_2.png"

# Sava image.
fig.savefig(fig_file, bbox_inches='tight')
fig_3d.savefig(fig_file_3d_1, bbox_inches='tight')
ax_3d.view_init(30,45)
fig_3d.savefig(fig_file_3d_2, bbox_inches='tight')

# Zoomed image.
ax.clear()
x_lim = 100
y_lim = 100
margin = 15 # for legend area
ax.set_title(title)
ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
ax.set_xlim(0, x_lim + margin)
ax.set_ylim(0, y_lim + margin)
draw(x_lim, y_lim, 'level')
ax.grid()
ax.legend()

fig_file = "result_box_zoom.png"
fig.savefig(fig_file, bbox_inches='tight')

# Distance image.
ax.clear()
ax.set_title(title)
ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
ax.set_xlim(0, x_lim + margin)
ax.set_ylim(0, y_lim + margin)
draw(x_lim, y_lim, 'dist')
ax.grid()
ax.legend()

fig_file = "result_box_dist.png"
fig.savefig(fig_file, bbox_inches='tight')

