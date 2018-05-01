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

def stab_invariant(image, box):
  border = 0.2
  max_width = 600
  im_height = len(image)
  im_width = len(image[0])
  (left, right, top, bottom) = (box[0], box[1], box[2], box[3])
  center = (right - left) // 2
  extra_sides = max_width // 2
  new_top = 0
  new_bottom = im_height
  new_left = 0 if (left - extra_sides) < 0 else (left - extra_sides)
  new_right = im_width if (right + extra_sides) > im_width else (right + extra_sides)
  output = image[new_top:new_bottom, new_left:new_right]
  return output

def stab_mock(image, box):
    return image

def detect_mock_init():
    return None

def detect_mock(net, frame):
    return None

def detect_res10_init():
    prototxt = 'caffe_ssd_res10.prototxt'
    caffemodel = 'caffe_ssd_res10.caffemodel'
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

def detect_fastrcnn_init():
    pb = 'tf_fastrcnn_inception.pb'
    pbtxt = 'tf_fastrcnn_inception.pbtxt'
    net = cv2.dnn.readNetFromTensorflow(pb, pbtxt)
    return net

def detect_fastrcnn(net, frame):
    inWidth = 300
    inHeight = 300
    means = (127.5, 127.5, 127.5)
    ratio = 1.0/127.5
    #net.setInput(dnn.blobFromImage(cv2.resize(frame, (inWidth, inHeight)), ratio, (inWidth, inHeight), means))
    net.setInput(dnn.blobFromImage(frame, ratio, (inWidth, inHeight), means, swapRB=True, crop=False))
    detections = net.forward()
    return detections

def detect_inception_openimages_init():
    pb = 'tf_ssd_inception_openimages.pb'
    pbtxt = 'tf_ssd_inception_openimages.pbtxt'
    net = cv2.dnn.readNetFromTensorflow(pb, pbtxt)
    return net

def detect_inception_openimages(net, frame):
    inWidth = 300
    inHeight = 300
    means = (127.5, 127.5, 127.5)
    ratio = 1.0/127.5
    #net.setInput(dnn.blobFromImage(cv2.resize(frame, (inWidth, inHeight)), ratio, (inWidth, inHeight), means))
    net.setInput(dnn.blobFromImage(frame, ratio, (inWidth, inHeight), means, swapRB=True, crop=False))
    detections = net.forward()
    return detections

def detect_inception_widerface_init():
    pb = 'tf_ssd_inception_widerface.pb'
    pbtxt = 'tf_ssd_inception_widerface.pbtxt'
    net = cv2.dnn.readNetFromTensorflow(pb, pbtxt)
    return net

def detect_inception_widerface(net, frame):
    inWidth = 300
    inHeight = 300
    means = (127.5, 127.5, 127.5)
    ratio = 1.0/127.5
    #net.setInput(dnn.blobFromImage(cv2.resize(frame, (inWidth, inHeight)), ratio, (inWidth, inHeight), means))
    net.setInput(dnn.blobFromImage(frame, ratio, (inWidth, inHeight), means, swapRB=True, crop=False))
    detections = net.forward()
    return detections

def detect_mobilenet_openimages_init():
    pb = "tf_ssd_mobilenet_openimages.pb"
    pbtxt = "tf_ssd_mobilenet_openimages.pbtxt"
    net = cv2.dnn.readNetFromTensorflow(pb, pbtxt)
    return net

def detect_mobilenet_openimages(net, frame):
    inWidth = 300
    inHeight = 300
    means = (127.5, 127.5, 127.5)
    ratio = 1.0/127.5
    #net.setInput(dnn.blobFromImage(cv2.resize(frame, (inWidth, inHeight)), ratio, (inWidth, inHeight), means))
    net.setInput(dnn.blobFromImage(frame, ratio, (inWidth, inHeight), means, swapRB=True, crop=False))
    detections = net.forward()
    return detections

def detect_mobilenet_widerface_init():
    pb = 'tf_ssd_mobilenet_openimages.pb'
    pbtxt = 'tf_ssd_mobilenet_openimages.pbtxt'
    net = cv2.dnn.readNetFromTensorflow(pb, pbtxt)
    return net

def detect_mobilenet_widerface(net, frame):
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

surf = cv2.xfeatures2d.SURF_create()
sift = cv2.xfeatures2d.SIFT_create()
desc = surf
last_descriptor = None

def stab_descriptor(image, bbox=None):
  global desc, last_descriptor
  if last_descriptor is None:
    desc_kp_1, desc_des_1 = desc.detectAndCompute(image, None)
  else:
    desc_kp_1, desc_des_1 = last_descriptor
  last_descriptor = desc.detectAndCompute(image, None)
  desc_kp_2, desc_des_2 = last_descriptor
  FLANN_INDEX_KDTREE = 1
  index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
  search_params = dict(checks=50)   # or pass empty dictionary
  flann = cv2.FlannBasedMatcher(index_params, search_params)
  matches = flann.knnMatch(desc_des_1, desc_des_2, k=2)
  MIN_MATCH_COUNT = 10
  good = []
  for m,n in matches:
    if m.distance < 0.7*n.distance:
      good.append(m)
  if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([ desc_kp_1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ desc_kp_2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    h,w,d = image.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    trans_coords = get_area_coords(dst)
    image = four_point_transform(image, trans_coords)
    #image = cv2.polylines(image,[np.int32(dst)],True,255,3, cv2.LINE_AA)
  return image

def get_area_coords(dst):
  tl = (int(dst[0][0][0]), int(dst[0][0][1]))
  tr = (int(dst[3][0][0]), int(dst[3][0][1]))
  bl = (int(dst[1][0][0]), int(dst[1][0][1]))
  br = (int(dst[2][0][0]), int(dst[2][0][1]))
  return [tl, tr, br, bl]

def order_points(pts):
  # initialzie a list of coordinates that will be ordered
  # such that the first entry in the list is the top-left,
  # the second entry is the top-right, the third is the
  # bottom-right, and the fourth is the bottom-left
  rect = np.zeros((4, 2), dtype = "float32")
 
  # the top-left point will have the smallest sum, whereas
  # the bottom-right point will have the largest sum
  s = pts.sum(axis = 1)
  rect[0] = pts[np.argmin(s)]
  rect[2] = pts[np.argmax(s)]
 
  # now, compute the difference between the points, the
  # top-right point will have the smallest difference,
  # whereas the bottom-left will have the largest difference
  diff = np.diff(pts, axis = 1)
  rect[1] = pts[np.argmin(diff)]
  rect[3] = pts[np.argmax(diff)]
 
  # return the ordered coordinates
  return rect

def four_point_transform(image, pts):
  # obtain a consistent order of the points and unpack them
  # individually
  pts = np.array(pts)
  rect = order_points(pts)
  # rect = np.array(pts)
  (tl, tr, br, bl) = rect
 
  # compute the width of the new image, which will be the
  # maximum distance between bottom-right and bottom-left
  # x-coordiates or the top-right and top-left x-coordinates
  widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
  widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
  maxWidth = max(int(widthA), int(widthB))
 
  # compute the height of the new image, which will be the
  # maximum distance between the top-right and bottom-right
  # y-coordinates or the top-left and bottom-left y-coordinates
  heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
  heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
  maxHeight = max(int(heightA), int(heightB))
 
  # now that we have the dimensions of the new image, construct
  # the set of destination points to obtain a "birds eye view",
  # (i.e. top-down view) of the image, again specifying points
  # in the top-left, top-right, bottom-right, and bottom-left
  # order
  dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype = "float32")
 
  # compute the perspective transform matrix and then apply it
  M = cv2.getPerspectiveTransform(rect, dst)
  warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
  # return the warped image
  return warped


if __name__ == '__main__':
    camera_number = 0
    write_file = False
    visualize = True
    use_tracking = False
    resize_image = None
    #resize_image = (888, 480)
    # stab_invariant stab_scale stab_resize
    stab_method = stab_scale
    # detect_mobilenet_widerface detect_mobilenet_openimages detect_inception_widerface detect_inception_openimages
    detect_method = detect_mobilenet_widerface
    detect_method_init = detect_mobilenet_openimages_init
    get_tracker = tracker_KCF

    net = detect_method_init()
    cap = cv2.VideoCapture(camera_number)
    use_detector = True
    ok = None
    out = None
    bbox = None
    lastFound = None
    prevFrameTime = None
    currentFrameTime = None
    font = cv2.FONT_HERSHEY_SIMPLEX
    avg = 0
    fps = 0
    num = 1
    size = 1
    weight = 2
    correct = 0
    time_det = 0
    time_sta = 0
    time_tra = 0
    accuracy = 0
    count_ms = 0
    count_fps = 0
    count_acc = 0
    count_det = 0
    count_sta = 0
    count_tra = 0
    frame_num = 0
    color = (255,255,255)

    if use_tracking:
      tracker = get_tracker()

    while True:
      start_time_total = time.time()
      frame_num += 1
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
          start_time = time.time()
          detections = detect_method(net, frame)
          time_det = (time.time() - start_time) * 1000
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
        else:
          if ok is None:
            ok = tracker.init(frame, bbox)
          else:
            start_time = time.time()
            ok, box = tracker.update(frame)
            time_tra = (time.time() - start_time) * 1000
            print('tracker: ', time_tra)
            box = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            if ok:
              bbox = box
              found = True
              box_color = (255, 0, 0)
            else:
              use_detector = True
              ok = None
        if found:
          correct += 1
          cv2.rectangle(frame, (bbox[0], bbox[2]), (bbox[1], bbox[3]), box_color)
        if bbox is not None:
          start_time = time.time()
          frame = stab_method(frame, bbox)
          time_sta = (time.time() - start_time) * 1000

      diff = time.time() - start_time_total
      ms = diff*1000

      fps = 1000 // ms
      accuracy = correct / frame_num
      count_fps += fps
      count_acc += accuracy * 100
      count_ms += diff * 1000
      count_det += time_det
      count_sta += time_sta
      count_tra += time_tra
      avg_ms = count_ms // num
      avg_fps = count_fps // num
      avg_acc = count_acc // num
      avg_det = count_det // num
      avg_sta = count_sta // num
      avg_tra = count_tra // num
      num += 1

      cv2.putText(frame, "fps: %s acc: %s ms: %s det: %s sta: %s tra: %s" % (1000//avg_ms, avg_acc, avg_ms, avg_det, avg_sta, avg_tra), (10, 30), font, size, color, weight)
      if write_file:
        out.write(frame)
      if visualize:
        cv2.imshow("detections", frame)
      else:
        print(avg_ms, 1000 // avg_ms)
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