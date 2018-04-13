import numpy as np
import argparse
import os
import sys
import time
import cv2
from cv2 import dnn

confThreshold = 0.5

def stab_scale(image, box):
  border = 0.2
  im_height = len(image)
  im_width = len(image[0])
  (left, right, top, bottom) = (box[0], box[1], box[2], box[3])
  border_height = (bottom - top) * border
  top = 0 if (top - border_height) < 0 else (top - border_height)
  bottom = im_height if (bottom + border_height) > im_height else (bottom + border_height)
  scale_y = im_height/(bottom - top)
  output = cv2.resize(image, (0,0), fy=scale_y, fx=scale_y)
  (xleft, xright, xtop, xbottom, xim_width) = (int(left*scale_y), int(right*scale_y),
                                              int(top*scale_y), int(bottom*scale_y),
                                              int(im_width*scale_y))
  extra_width = (im_width - (xright - xleft)) // 2
  new_left = 0 if (xleft - extra_width) < 0 else (xleft - extra_width)
  new_right = xim_width if (xright + extra_width) > xim_width else (xright + extra_width)
  output = output[xtop:xbottom, new_left:new_right]
  return output

def stab_resize(image, box):
  border = 0.2
  im_height = len(image)
  im_width = len(image[0])
  (left, right, top, bottom) = (box[0], box[1], box[2], box[3])
  border_height = (bottom - top) * border
  top = int(0 if (top - border_height) < 0 else (top - border_height))
  bottom = int(im_height if (bottom + border_height) > im_height else (bottom + border_height))
  scale_y = im_height/(bottom - top)
  scale_x = (bottom - top) / im_height
  new_width = im_width * scale_x
  extra_width = (new_width - (right - left)) // 2
  new_left = int(0 if (left - extra_width) < 0 else (left - extra_width))
  new_right = int(im_width if (right + extra_width) > im_width else (right + extra_width))
  output = cv2.resize(image[top:bottom, new_left:new_right], (0,0), fy=scale_y, fx=scale_y)
  return output

def stab_mock(image, box):
    return image

def detect_mock_init():
    return None

def detect_mock(net, frame):
    return None

def detect_res10_init():
    prototxt = 'res10_300x300_ssd_iter_140000.prototxt'
    caffemodel = 'res10_300x300_ssd_iter_140000.caffemodel'
    net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
    return net

def detect_res10(net, frame):
    inWidth = 300
    inHeight = 300
    net.setInput(dnn.blobFromImage(cv2.resize(frame, (inWidth, inHeight)), 1.0, (inWidth, inHeight), (104., 177., 123.)))
    detections = net.forward()
    return detections

if __name__ == '__main__':
    camera_number = 0
    write_file = False
    visualize = True
    #resize_image = None
    resize_image = (888, 480)
    stab_method = stab_scale
    detect_method = detect_res10
    detect_method_init = detect_res10_init

    net = detect_method_init()
    cap = cv2.VideoCapture(camera_number)
    font = cv2.FONT_HERSHEY_SIMPLEX
    lastFound = None
    prevFrameTime = None
    currentFrameTime = None
    size = 1
    color = (255,255,255)
    weight = 2
    avg = 0
    fps = 0
    num = 1
    cont = 0
    out = None

    while True:
        ret, frame = cap.read()
        if resize_image is not None:
          frame = cv2.resize(frame, resize_image)
        cols = frame.shape[1]
        rows = frame.shape[0]
        if write_file and out is None:
          out = cv2.VideoWriter("out.avi", cv2.VideoWriter_fourcc(*'H264'), 25.0, (cols, rows))
        
        if net:
          detections = detect_method(net, frame)

          """
          perf_stats = net.getPerfProfile()
          print('Inference time, ms: %.2f' % (perf_stats[0] / cv2.getTickFrequency() * 1000))
          """
          found = False
          for i in range(detections.shape[2]):
              confidence = detections[0, 0, i, 2]
              if confidence > confThreshold:
                  found = True
                  xLeftBottom = int(detections[0, 0, i, 3] * cols)
                  yLeftBottom = int(detections[0, 0, i, 4] * rows)
                  xRightTop = int(detections[0, 0, i, 5] * cols)
                  yRightTop = int(detections[0, 0, i, 6] * rows)

                  cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop), (0, 255, 0))
                  #label = "face: %.4f" % confidence
                  #labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                  #cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]), (xLeftBottom + labelSize[0], yLeftBottom + baseLine), (255, 255, 255), cv2.FILLED)
                  #cv2.putText(frame, label, (xLeftBottom, yLeftBottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                  lastFound = (xLeftBottom, xRightTop, yLeftBottom, yRightTop)
                  frame = stab_method(frame, lastFound)
          if not found and lastFound is not None:
              frame = stab_method(frame, lastFound)

        prevFrameTime = currentFrameTime
        currentFrameTime = time.time()
        if (prevFrameTime != None):
            diff = currentFrameTime - prevFrameTime
            fps = 1.0 / diff
            cont += fps
            avg = cont // num
            num += 1
        cv2.putText(frame, str(avg), (10, 30), font, size, color, weight)
        if write_file:
          out.write(frame)
        if visualize:
          cv2.imshow("detections", frame)
        else:
          print(avg)
        if cv2.waitKey(1) != -1:
            break

#no vis + no face + no stab + no save = 32fps
#no vis + no face + no stab + save = 32fps
#vis + no face + no stab + no save = 15fps

#res10
#vis + face = 9fps
#vis + face + stab = 7fps
#vis + face + stab + save = 6fps
#no vis + face + stab + save = 15fps