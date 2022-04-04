import sys
import numpy as np

def calcIoU(a, g):
  min_x=min(a.x1, a.x2, g.x1, g.x2)
  max_x=max(a.x1, a.x2, g.x1, g.x2)
  min_y=min(a.y1, a.y2, g.y1, g.y2)
  max_y=max(a.y1, a.y2, g.y1, g.y2)

  if a.x2-a.x1 + g.x2-g.x1 <= max_x-min_x or a.y2-a.y1 + g.y2-g.y1 <= max_y-min_y :
    return 0
  else:
    intersection = (sorted([a.x1, a.x2, g.x1, g.x2])[2]- sorted([a.x1, a.x2, g.x1, g.x2])[1]) * (sorted([a.y1, a.y2, g.y1, g.y2])[2]- sorted([a.y1, a.y2, g.y1, g.y2])[1])
    union = (a.x2-a.x1)*(a.y2-a.y1) + (g.x2-g.x1)*(g.y2-g.y1) - intersection
    #print(intersection, union, ": ", intersection/union)
    iou=intersection/union
    return round(iou, 2)

class Point:
  x1=0
  y1=0
  x2=0
  y2=0

result=[]
f = open("../model/out/kitti.trainval", 'r')
line = f.readline()

box_count = 0
while True:
  if not line: break
  if line[0]=="#":
    line = f.readline()
    image_bbox=[]

    while line[0]!="#":
      line = f.readline()
      if line[0:2] =="7 ":
        point=Point()
        point.x1=float(line.split()[1])
        point.y1=float(line.split()[2])
        point.x2=float(line.split()[3])
        point.y2=float(line.split()[4])
        image_bbox.append(point)
        box_count+=1
      if not line: break
    for i in range(0, len(image_bbox)-1):
      for j in range(i+1, len(image_bbox)):
        iou=calcIoU(image_bbox[i], image_bbox[j])
        if iou != 0:
          result.append(iou)
f.close()

result_np = np.array(result)

print("Total box count: %d" % box_count)
print("Total overlap count: %d" % result_np.shape)

for i in range(0, 10):
  iou = float(i)/10
  print("[IoU %.2f] %d" % (iou, np.count_nonzero( result_np > iou )) )

