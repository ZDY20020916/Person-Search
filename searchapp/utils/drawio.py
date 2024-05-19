import cv2
import os


def drawbox(img_path, detections):
    image = cv2.imread(img_path)
    x1, y1, x2, y2, conf = detections[0], detections[1], detections[2], detections[3], detections[4]

