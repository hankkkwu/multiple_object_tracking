"""
# 1 - Import Libraries and Test Images
"""

### Imports
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import copy
import pickle
import cv2
import numpy as np
from yolo_nms import *
import time

### Load the Images
# dataset_images = pickle.load(open('./Images/images_tracking.p', "rb"))

def visualize_images(input_images):
    fig=plt.figure(figsize=(100,100))

    for i in range(len(input_images)):
        fig.add_subplot(1, len(input_images), i+1)
        plt.imshow(input_images[i])
    plt.show()

# visualize_images(dataset_images)


def id_to_color(idx):
    """
    Random function to convert an id to a color
    Do what you want here but keep numbers below 255
    """
    blue = idx*5 % 256
    green = idx*36 %256
    red = idx*23 %256
    return (red, green, blue)

"""
# 2 - Association
"""

def convert_data(box):
    """
    Convert data from (x1,y1, w, h) to (x1,y1,x2,y2)
    """
    x1 = int(box[0])
    x2 = int(box[0] + box[2])
    y1 = int(box[1])
    y2 = int(box[1]+box[3])
    return x1,y1,x2,y2

def box_iou(box1, box2):
    """
    Compute Intersection Over Union cost
    """
    box1 = convert_data(box1)
    box2 = convert_data(box2)
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1) #abs((xi2 - xi1)*(yi2 - yi1))
    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1) #abs((box1[3] - box1[1])*(box1[2]- box1[0]))
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1) #abs((box2[3] - box2[1])*(box2[2]- box2[0]))
    union_area = (box1_area + box2_area) - inter_area
    # compute the IoU
    iou = inter_area/float(union_area)
    return iou


## CREATE A FUNCTION HUNGARIAN_COST THAT OUTPUTS THE HUNGARIAN COST AS IN THE PAPER (https://arxiv.org/pdf/1709.03572.pdf)
## BE CAREFUL NOT TO DIVIDE BY ZERO
def c_lin(XA, YA, WA, HA, XB, YB, WB, HB):
  '''
  calculate the linear cost
  '''
  # (_ , w, h, _) = np.array(dataset_images).shape
  w = 960 # 960  # 3840
  h = 540 # 540  # 2160
  Q_dist = np.linalg.norm(np.array([w, h]))
  Q_shp = w * h
  d1 = np.linalg.norm(np.array([XA-XB, YA-YB]))
  d2 = np.linalg.norm(np.array([HA-HB, WA-WB]))

  if d1 == 0.0:
    d1 = 0.0001
  if d2 == 0.0:
    d2 = 0.0001
  cost = (Q_dist / d1) * (Q_shp / d2)
  return cost

def c_exp(XA, YA, WA, HA, XB, YB, WB, HB):
  '''
  calculate the exponential cost
  '''
  w1 = 0.5
  w2 = 1.5
  p1 = ((XA-XB) / WA)**2 + ((YA-YB) / HA)**2
  p2 = abs(HA - HB) / (HA + HB) + abs(WA - WB) / (WA + WB)
  cost = np.exp(-w1 * p1) * np.exp(-w2 * p2)
  return cost


## YOUR CODE HERE
def hungarian_cost(old_boxes, new_boxes, iou_thresh=0.3, linear_thresh=10000, exp_thresh=0.5):
    """
    Combine the IOU costs, Exponential costs, and linear costs.
    Those thresholds are the suggestions in the paper.
    RETURN a cost matrix.
    """
    cost_matrix = []
    for box1 in old_boxes:
      row = []
      for box2 in new_boxes:
        iou_cost = box_iou(box1, box2)
        XA = box1[0] + box1[2] * 0.5
        YA = box1[1] + box1[3] * 0.5
        WA = box1[2]
        HA = box1[3]
        XB = box2[0] + box2[2] * 0.5
        YB = box2[1] + box2[3] * 0.5
        WB = box2[2]
        HB = box2[3]
        # XA = (box1[2] - box1[0]) * 0.5
        # YA = (box1[3] - box1[1]) * 0.5
        # WA = box1[2] - box1[0]
        # HA = box1[3] - box1[1]
        # XB = (box2[2] - box2[0]) * 0.5
        # YB = (box2[3] - box2[1]) * 0.5
        # WB = box2[2] - box2[0]
        # HB = box2[3] - box2[1]
        lin_cost = c_lin(XA, YA, WA, HA, XB, YB, WB, HB)
        exp_cost = c_exp(XA, YA, WA, HA, XB, YB, WB, HB)

        if (iou_cost >= iou_thresh and lin_cost >= linear_thresh and exp_cost >= exp_thresh):
          row.append(iou_cost)
        else:
          row.append(0)
      cost_matrix.append(row)

    return cost_matrix


"""
# 3 - The Hungarian Algorithm
"""

from scipy.optimize import linear_sum_assignment

def associate(old_boxes, new_boxes):
    """
    old_boxes will represent the former bounding boxes (at time 0)
    new_boxes will represent the new bounding boxes (at time 1)
    Function goal: Define a Hungarian Matrix with IOU as a metric and return, for each box, an id
    RETURN: Matches, Unmatched Detections, Unmatched Trackers
    """
    ## YOUR CODE HERE
    iou_matrix = np.array(hungarian_cost(old_boxes, new_boxes))

    # We are having the maximization problem, we want to take the maximum cost.
    # But the Hungarian algorithm works with minimization problem. It takes
    # the minimum distance, or minimum numbers in the matrix
    hungarian_row, hungarian_col = linear_sum_assignment(-iou_matrix)

    matches = []
    unmatched_trackers = []
    unmatched_detections = []

    for i in range(len(hungarian_row)):
      x = hungarian_row[i]
      y = hungarian_col[i]
      if iou_matrix[x][y] < 0.3:
        unmatched_trackers.append(x)
        unmatched_detections.append(new_boxes[y])
      else:
        matches.append([x,y])

    # Since the number of old_boxes and the number of new_boxes is not always the same,
    # but the length of hungarian_row and hungarian_col will be the same,
    # so there might be some of the untracked or new_detected not been append to the list
    for t, track in enumerate(old_boxes):
      if t not in hungarian_row:
        unmatched_trackers.append(t)

    for d, det in enumerate(new_boxes):
      if d not in hungarian_col:
        unmatched_detections.append(det)

    return matches, unmatched_trackers, unmatched_detections


"""
# 4 - Using Age
A false positive means that you detected an obstacle that shouldn't detect.
We'll solve it by introducing a MIN_HIT_STREAK variable. If the detector detects something once, it is not displayed. If it detects it twice in a row, or 3 times in a row (thanks to matching), it is displayed.

A false negative means that you didn't detect an obstacle that should have been detected.
We'll solve it by introducing a MAX_AGE variable. If an obstacle is suddently unmatched, we keep displaying it. If it is unmatched again, or more times, we remove it.
"""

MIN_HIT_STREAK = 3
MAX_UNMATCHED_AGE = 7

"""
Define the Obstacle class to include these values
Every obstacle should have:
* an id
* a box
* current time
* an age (number of times matched)
* an unmatched frame number (number of times unmatched)
* kalman filter filter for every object
"""

from filterpy.kalman import KalmanFilter

class Obstacle():
    def __init__(self, idx, box, time, age=1, unmatched_age=0):
        """
        box = [x1, y1, w, h]
        """
        self.idx = idx
        self.box = box
        self.time = time
        self.age = age
        self.unmatched_age = unmatched_age

        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        self.kf.x = np.array([self.box[0], 0, self.box[1], 0, self.box[2], 0, self.box[3], 0])
        self.kf.P *= 1000
        Q_std = 0.01
        self.kf.Q[4:, 4:] *= Q_std

        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0]])
        R_std = 10
        self.kf.R[2:, 2:] *= R_std

def get_box_from_state(state):
    """
    get bounding box info from x state
    """
    return [state[0], state[2], state[4], state[6]]

def return_F_with_dt(dt):
    """
    Retrun F matrix that include delta t
    dt should be the time between image frames
    """
    # for images from "images_tracking.p", the image was taken every 7 frames, and the FPS is 60
    # dt = 7 / 60
    # for video from MOT16, the FPS is 25
    dt = 1 / 25
    # for video from paris_challenge, the FPS is 60
    # dt = 1 / 60

    F = np.eye(8)
    F[0, 1] = dt
    F[2, 3] = dt
    F[4, 5] = dt
    F[6, 7] = dt
    return F


"""
# 5 - Main Loop
"""

def main(input_image):
    """
    Receives an images
    Outputs the result image, and a list of obstacle objects
    """
    ## MODIFY THE MAIN FUNCTION TO NOW CONSIDER AGE AND UNMATCHED AGE
    global stored_obstacles
    global idx
    global yolo

    image = copy.deepcopy(input_image)
    _, bounding_boxes = yolo.inference(image)
    current_time = time.time()

    # see if this is the first
    # since we start the idx from 0, if this is the first image, idx will be 0
    if idx == 0:
      stored_obstacles = []
      for box in bounding_boxes:
        # assign idx to each box
        obj = Obstacle(idx, box, current_time)
        stored_obstacles.append(obj)
        # draw the bounding box on image
        # left, top, right, bottom = convert_data(box)
        # color = id_to_color(idx)
        # image = cv2.rectangle(image, (left, top), (right, bottom), color, 7)
        # display the idx at the top of the bounding box
        # text = "{}".format(idx)
        # cv2.putText(image, text, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
        idx += 1
      return image

    else:
      # get previous frame boxes
      previous_boxes = [obj.box for obj in stored_obstacles]
      # do association between new detected boxes and previous detected boxes
      matches, unmatched_trackers, unmatched_detections = associate(previous_boxes, bounding_boxes)

      new_obstacles = []
      selected_obstacles = []   # this is gonna work with condition
      """
      For matched boxes, store it to the new_obstacles list
      matches is an array store matched boxes index [[index of pre_boxes, index of cur_boxes], [1,2] , [2,3]...]
      """
      for match in matches:
        # take the former obstacle and its ID, increment the age by 1
        obj = stored_obstacles[match[0]]   # Obstacle(stored_obstacles[match[0]].idx, bounding_boxes[match[1]], stored_obstacles[match[0]].age+1)
        obj.age += 1
        obj.unmatched_age = 0

        # Update
        measurement = np.array(bounding_boxes[match[1]])
        obj.kf.update(measurement)

        # Prediction
        # calculate dt
        dt = current_time - obj.time
        obj.kf.F = return_F_with_dt(dt)
        obj.kf.predict()

        # Update for future match
        obj.time = current_time
        obj.box = get_box_from_state(obj.kf.x)
        new_obstacles.append(obj)

        if obj.age >= MIN_HIT_STREAK:
          selected_obstacles.append(obj)

      """
      For unmatched detections
      """
      for new_box in unmatched_detections:
        # if we have new detected obstacles, assign an idx to it
        obj = Obstacle(idx, new_box, current_time)
        new_obstacles.append(obj)
        idx += 1

      """
      For unmatched tracks, do prediction
      """
      for index in unmatched_trackers:
        obj = stored_obstacles[index]
        obj.unmatched_age += 1

        dt = current_time - obj.time
        obj.kf.F = return_F_with_dt(dt)
        obj.kf.predict()

        # Update for future match
        obj.time = current_time
        obj.box = get_box_from_state(obj.kf.x)

        if obj.unmatched_age < MAX_UNMATCHED_AGE:
          selected_obstacles.append(obj)
          new_obstacles.append(obj)

      # draw bounding boxes on image
      for obj in selected_obstacles:
        new_idx = obj.idx
        box = obj.box
        left, top, right, bottom = convert_data(box)
        color = id_to_color(new_idx)
        image = cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), color, 2)
        text = "{}".format(new_idx)
        image = cv2.putText(image, text, (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

      stored_obstacles = copy.deepcopy(new_obstacles)
      return image



yolo = YOLO()
idx = 0

# fig=plt.figure(figsize=(100,100))
test_image = mpimg.imread('./Images/test1.JPG')
# result_images_3 = copy.deepcopy(dataset_images)

#out_img = main(test_image)
#plt.imshow(out_img)
"""
out_imgs = []
for i in range(len(result_images_3)):
    out_img = main(result_images_3[i])
    out_imgs.append(out_img)
    fig.add_subplot(1, len(result_images_3), i+1)
    plt.imshow(out_imgs[i])
"""

# plt.show()



"""
# 6 - Process Video
"""
# Commented out IPython magic to ensure Python compatibility.
from moviepy.editor import VideoFileClip
# video_file = "./Images/paris_challenge.mov"
#video_file = "./Images/MOT16-13-raw.mp4" #25 FPS
video_file = "./Images/MOT16-14-raw.mp4" #25 FPS

clip = VideoFileClip(video_file)
white_clip = clip.fl_image(main)
white_clip.write_videofile("./Images/MOT16-14-raw_kf_out.mp4",audio=False)
