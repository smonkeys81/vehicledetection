# Copyright 2020 Robotics, Inc. All Rights Reserved.

# This code reads loss values from file and display loss in image.

from matplotlib import pyplot as plt
import numpy as np
import argparse
import os
import matplotlib.animation as anim
from matplotlib.widgets import Button

# Parse arguements.
parser = argparse.ArgumentParser(description='Plot loss values.')
parser.add_argument('--file', dest='file_name', help='Path to file', default=None, type=str)
args = parser.parse_args()

# Loss data from caffe generated file
# it, loss, smt_loss, loss_bbox, loss_cls, rpn_loss_cls, rpn_loss_bbox 
data = np.loadtxt(args.file_name, delimiter=",")

# Get iteration size
max_iter = int(np.max(data[:, 1]) + 1)

# Format plots
# Number of column in plot
num_col = max_iter * 2
# Figure and axes object
fig, axes = plt.subplots(4, num_col, sharex=True, sharey=True, figsize=(max_iter * 6, 13) )
fig.suptitle('Vehicle Detection Pipeline Alpha - Loss Monitor', size = 10 + max_iter * 5)

# Initial Y axis limit of smoothed loss
init_ylim_smooth = 8
# Initial Y axis limit of loss at the end
init_ylim_end = 1
# Initial Y axis limit of loss at the RPN
init_ylim_rpn = 1
# Angle of iteration characters
init_xtick_degree = 45

# Deployment of subplots.
axes = []
# smoothed loss.
ax = plt.subplot(4, 1, 1)
ax.set_ylim(0, init_ylim_smooth)
ax.grid()
ax.set_title('smoothed loss')
axes.append(ax)
for i in range(max_iter):
  plt.subplots_adjust(wspace = 0.03, hspace = 0.3)

  # loss.
  ax = plt.subplot(4, max_iter, max_iter + i + 1) 
  if i != 0:
    ax.set_yticklabels([])
  plt.xticks(rotation=init_xtick_degree)
  ax.grid()
  ax.set_ylim(0, init_ylim_smooth)
  ax.set_title("loss %d" % (i+1))
  axes.append(ax)
  # loss-bbox.
  ax = plt.subplot(4, max_iter*2, max_iter*4 + 1 + i*2)
  ax.set_xticklabels([])
  if i != 0:
    ax.set_yticklabels([])
  ax.grid()
  ax.set_ylim(0, init_ylim_end)
  ax.set_title("loss_bbox %d" % (i+1))
  axes.append(ax)
  # loss-cls.
  ax = plt.subplot(4, max_iter*2, max_iter*4 + 2 + i*2)
  ax.set_xticklabels([])
  ax.set_yticklabels([])
  ax.grid()
  ax.set_ylim(0, init_ylim_end)
  ax.set_title("loss_cls %d" % (i+1))
  axes.append(ax)
  # loss-rpn-bbox.
  ax = plt.subplot(4, max_iter*2, max_iter*6 + 1 + i*2)
  if i != 0:
    ax.set_yticklabels([])
  plt.xticks(rotation=init_xtick_degree)
  ax.grid()
  ax.set_ylim(0, init_ylim_rpn)
  ax.set_title("RPN_loss_bbox %d" % (i+1))
  axes.append(ax)
  # loss-rpn-cls.
  ax = plt.subplot(4, max_iter*2, max_iter*6 + 2 + i*2)
  ax.set_yticklabels([])
  plt.xticks(rotation=init_xtick_degree)
  ax.grid()
  ax.set_ylim(0, init_ylim_rpn)
  ax.set_title("RPN_loss_cls %d" % (i+1))
  axes.append(ax)

def magnify(axis, zoomin):
  """Adjust limit of y axis for magnification

  :param axis Plot number to update
  :param zoomin To zoom in or zoom out
  """
  # Small number to figure out the range
  err = 0.001
  # Min and Max of current limit
  ymin, ymax = axes[axis].get_ylim()
  if ymax >= 2 or ymax <= (0.2 + err):
    if zoomin == True:
      ymax = ymax / 2
    elif zoomin == False:
      ymax = ymax * 2
  else:
    if zoomin == True:
      ymax = ymax - 0.2
    elif zoomin == False:
      ymax = ymax + 0.2
  
  axes[axis].set_ylim([0, ymax])
  

# Update function by buttons.
class Index(object):
  """Button event
  """
  def zoomin_smt(self, event):
    """ZoomIn button of smoothed loss plot
    """
    magnify(0, True)
  def zoomout_smt(self, event):
    """ZoomOut button of smoothed loss plot
    """
    magnify(0, False)

  def zoomin_loss(self, event):
    """ZoomIn button of loss plot
    """
    for i in range(max_iter):
      magnify(i * 5 + 1, True)
  def zoomout_loss(self, event):
    """ZoomOut button of loss plot
    """
    for i in range(max_iter):
      magnify(i * 5 + 1, False)

  def zoomin_loss_end(self, event):
    """ZoomIn button of losses at the end
    """
    for i in range(max_iter):
      magnify(i * 5 + 2, True)
      magnify(i * 5 + 3, True)
  def zoomout_loss_end(self, event):
    """ZoomOut button of losses at the end
    """
    for i in range(max_iter):
      magnify(i * 5 + 2, False)
      magnify(i * 5 + 3, False)

  def zoomin_loss_rpn(self, event):
    """ZoomIn button of losses at the RPN
    """
    for i in range(max_iter):
      magnify(i * 5 + 4, True)
      magnify(i * 5 + 5, True)
  def zoomout_loss_rpn(self, event):
    """ZoomOut button of losses at the RPN
    """
    for i in range(max_iter):
      magnify(i * 5 + 4, False)
      magnify(i * 5 + 5, False)

# Button string - zoom in
str_zin = "Zoom\n(+)"
# Button string - zoom out
str_zout = "Zoom\n(-)"

callback = Index()

# Button size and position - width
button_width = 0.07
# Button size and position - height
button_height = 0.05
# Button size and position - margin from the left
button_left = 0.025
# Button size and position - margin from the top
button_y_zin = 0.8
# Button size and position - margin from the top
button_y_zout = 0.725


# Create buttons and events.
# smooth.
ax_zin_smt = plt.axes([button_left, button_y_zin, button_width, button_height])
ax_zout_smt = plt.axes([button_left, button_y_zout, button_width, button_height])
bzoomin_smt = Button(ax_zin_smt, str_zin)
bzoomout_smt = Button(ax_zout_smt, str_zout)
bzoomin_smt.on_clicked(callback.zoomin_smt)
bzoomout_smt.on_clicked(callback.zoomout_smt)
button_y_zin = button_y_zin - 0.21
button_y_zout = button_y_zout - 0.21
# total loss.
ax_zin_loss = plt.axes([button_left, button_y_zin, button_width, button_height])
ax_zout_loss = plt.axes([button_left, button_y_zout, button_width, button_height])
bzoomin_loss = Button(ax_zin_loss, str_zin)
bzoomout_loss = Button(ax_zout_loss, str_zout)
bzoomin_loss.on_clicked(callback.zoomin_loss)
bzoomout_loss.on_clicked(callback.zoomout_loss)
button_y_zin = button_y_zin - 0.21
button_y_zout = button_y_zout - 0.21
# loss-end.
ax_zin_loss_end = plt.axes([button_left, button_y_zin, button_width, button_height])
ax_zout_loss_end = plt.axes([button_left, button_y_zout, button_width, button_height])
bzoomin_loss_end = Button(ax_zin_loss_end, str_zin)
bzoomout_loss_end = Button(ax_zout_loss_end, str_zout)
bzoomin_loss_end.on_clicked(callback.zoomin_loss_end)
bzoomout_loss_end.on_clicked(callback.zoomout_loss_end)
button_y_zin = button_y_zin - 0.21
button_y_zout = button_y_zout - 0.21
# loss-rpn.
ax_zin_loss_rpn = plt.axes([button_left, button_y_zin, button_width, button_height])
ax_zout_loss_rpn = plt.axes([button_left, button_y_zout, button_width, button_height])
bzoomin_loss_rpn = Button(ax_zin_loss_rpn, str_zin)
bzoomout_loss_rpn = Button(ax_zout_loss_rpn, str_zout)
bzoomin_loss_rpn.on_clicked(callback.zoomin_loss_rpn)
bzoomout_loss_rpn.on_clicked(callback.zoomout_loss_rpn)



# Threshold of iteration to determine long term
long_term = 500
# Display range when training becomes long term
display = 500
def update(frame):
  """Load data and update plot
  """
  #it, loss, smt_loss, loss_bbox, loss_cls, rpn_loss_cls, rpn_loss_bbox 
  data = np.loadtxt(args.file_name, delimiter=",")
  
  # Start index of raw data
  idx_start = 0

  length = int(data.shape[0]/max_iter)
  if length > long_term:
    # End limit of axis
    end = data[data.shape[0]-1, 0]
    # Start limit of axis
    start = end - display
    axes[0].set_xlim([start, end])
    for i in range(max_iter*5):
      axes[i+1].set_xlim([start, end])
    # Calculate start idx.
    idx_start = int(data.shape[0]-1 -(display * max_iter))

  # Update smoothed loss.
  axes[0].plot(data[idx_start:, 0], data[idx_start:, 2], 'g-')

  for i in range(max_iter):
    # Plot index
    idx = i * 5 + 1
    # Data index of start
    idx_s = idx_start + i

    # loss.
    axes[idx].plot(data[idx_s::max_iter, 0], data[idx_s::max_iter, 3], 'r-')
    # loss-bbox.
    axes[idx+1].plot(data[idx_s::max_iter, 0], data[idx_s::max_iter, 4], 'b-')
    # loss-cls.
    axes[idx+2].plot(data[idx_s::max_iter, 0], data[idx_s::max_iter, 5], 'c-')
    # loss-rpn-bbox.
    axes[idx+3].plot(data[idx_s::max_iter, 0], data[idx_s::max_iter, 7], 'y-')
    # loss-rpn-cls.
    axes[idx+4].plot(data[idx_s::max_iter, 0], data[idx_s::max_iter, 6], 'm-')

  return

ani = anim.FuncAnimation(fig, update, interval=500)

# Show image.
plt.show()
