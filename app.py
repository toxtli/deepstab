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
    means = (104., 177., 123.)
    ratio = 1.0
    #net.setInput(dnn.blobFromImage(cv2.resize(frame, (inWidth, inHeight)), ratio, (inWidth, inHeight), means))
    net.setInput(dnn.blobFromImage(frame, ratio, (inWidth, inHeight), means, swapRB=True, crop=False))
    detections = net.forward()
    return detections

def detect_inference_init():
    pb = 'inference.pb'
    pbtxt = 'inference.pbtxt'
    net = cv2.dnn.readNetFromTensorflow(pb, pbtxt)
    return net

def detect_inference(net, frame):
    inWidth = 300
    inHeight = 300
    means = (127.5, 127.5, 127.5)
    ratio = 1.0/127.5
    #net.setInput(dnn.blobFromImage(cv2.resize(frame, (inWidth, inHeight)), ratio, (inWidth, inHeight), means))
    net.setInput(dnn.blobFromImage(frame, ratio, (inWidth, inHeight), means, swapRB=True, crop=False))
    detections = net.forward()
    return detections

def tracker_KCF():
    return cv2.TrackerKCF_create()

def tracker_MedianFlow():
    return cv2.TrackerMedianFlow_create()

def tracker_Boosting():
    return cv2.TrackerBoosting_create()

def tracker_MIL():
    return cv2.TrackerMIL_create()

def tracker_TLD():
    return cv2.TrackerTLD_create()

def tracker_GOTURN():
    return cv2.TrackerGOTURN_create()

if __name__ == '__main__':
    camera_number = 0
    write_file = False
    visualize = True
    use_tracking = True
    #resize_image = None
    resize_image = (888, 480)
    stab_method = stab_scale
    detect_method = detect_res10
    detect_method_init = detect_res10_init
    get_tracker = tracker_KCF

    net = detect_method_init()
    cap = cv2.VideoCapture(camera_number)
    use_detector = True
    ok = None
    bbox = None
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

    if use_tracking:
      tracker = get_tracker()

    while True:
        ret, frame = cap.read()
        if resize_image is not None:
          frame = cv2.resize(frame, resize_image)
        cols = frame.shape[1]
        rows = frame.shape[0]
        if write_file and out is None:
          out = cv2.VideoWriter("out.avi", cv2.VideoWriter_fourcc(*'H264'), 25.0, (cols, rows))
        
        if net:
          found = False
          if not use_tracking or bbox is None or use_detector:
            detections = detect_method(net, frame)
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > confThreshold:
                    found = True
                    use_detector = False
                    xLeftBottom = int(detections[0, 0, i, 3] * cols)
                    yLeftBottom = int(detections[0, 0, i, 4] * rows)
                    xRightTop = int(detections[0, 0, i, 5] * cols)
                    yRightTop = int(detections[0, 0, i, 6] * rows)
                    bbox = (xLeftBottom, xRightTop, yLeftBottom, yRightTop)
                    box_color = (0, 255, 0)
                    #print(bbox)
          else:
            if ok is None:
              ok = tracker.init(frame, bbox)
            else:
              ok, box = tracker.update(frame)
              box = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
              if ok:
                found = True
                bbox = box
                box_color = (255, 0, 0)
                #print(bbox)
              else:
                use_detector = True
          if found:
            cv2.rectangle(frame, (bbox[0], bbox[2]), (bbox[1], bbox[3]), box_color)
          if bbox is not None:
            frame = stab_method(frame, bbox)

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