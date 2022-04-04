# Copyright 2019 Robotics, Inc. All Rights Reserved.

# This code reads loss values from file and save plot image.

from matplotlib import pyplot as plt
import numpy as np
import argparse
import os

# Parse arguements.
parser = argparse.ArgumentParser(description='Plot loss values.')
parser.add_argument('--file', dest='file_name', help='Path to file', default=None, type=str)
args = parser.parse_args()

#it, loss, smt_loss, loss_bbox, loss_cls, rpn_loss_cls, rpn_loss_bbox 
data = np.loadtxt(args.file_name, delimiter=",")

# Get iteration size.
max_iter = int(np.max(data[:, 1]) + 1)

# Format plots
fig = plt.figure(figsize=(max_iter*8, 16))
fig.suptitle('Vehicle Detection Pipeline Alpha', size = 20)

num_col = max_iter*2

# Plot data.

# smoothed loss.
plt.subplot(4, 1, 1)
plt.plot(data[:, 0], data[:, 2], 'g-')
plt.grid(b=True, which='both', axis='both')
plt.ylim(0, 2)
plt.title('smoothed loss')

for i in range(max_iter):
  plt.subplot(4, max_iter, max_iter + i + 1)
  plt.plot(data[i::max_iter, 0], data[i::max_iter, 3], 'r-')
  plt.grid(b=True, which='both', axis='both')
  plt.ylim(0, 1.5)  
  title = "loss %d" % (i+1)
  plt.title(title)

  plt.subplot(4, max_iter*2, max_iter*4 + 1 + i*2)
  plt.plot(data[i::max_iter, 0], data[i::max_iter, 4], 'b-')
  plt.grid(b=True, which='both', axis='both')
  plt.ylim(0, 1)
  title = "loss_bbox %d" % (i+1)
  plt.title(title)

  plt.subplot(4, max_iter*2, max_iter*4 + 2 + i*2)
  plt.plot(data[i::max_iter, 0], data[i::max_iter, 5], 'c-')
  plt.grid(b=True, which='both', axis='both')
  title = "loss_cls %d" % (i+1)
  plt.ylim(0, 1)
  plt.title(title)

  plt.subplot(4, max_iter*2, max_iter*6 + 1 + i*2)
  plt.plot(data[i::max_iter, 0], data[i::max_iter, 7], 'y-')
  plt.grid(b=True, which='both', axis='both')
  plt.ylim(0, 1)
  title = "rpn_loss_bbox %d" % (i+1)
  plt.title(title)

  plt.subplot(4, max_iter*2, max_iter*6 + 2 + i*2)
  plt.plot(data[i::max_iter, 0], data[i::max_iter, 6], 'm-')
  plt.grid(b=True, which='both', axis='both')
  plt.ylim(0, 1)
  title = "rpn_loss_cls %d" % (i+1)
  plt.title(title)

# Generate file name.
fig_file = os.path.splitext(args.file_name)[0]
fig_file = fig_file + ".png"

# Sava image.
plt.savefig(fig_file, bbox_inches='tight')
