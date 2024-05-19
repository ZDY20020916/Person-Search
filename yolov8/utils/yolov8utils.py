from PIL import Image, ImageDraw
import numpy as np


classes_names_coco = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
                     10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
                     20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
                     30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
                     40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
                     50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
                     60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
                     70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}


def _xywh_to_xyxy(bbox_xywh, width, height):
    x, y, w, h = bbox_xywh
    x1 = max(int(x-w/2), 0)
    x2 = min(int(x+w/2), width-1)
    y1 = max(int(y-h/2), 0)
    y2 = min(int(y+h/2), height-1)
    return x1, y1, x2, y2


def drawDetectBox(img, ret):
    draw = ImageDraw.Draw(img)
    # 绘制所有检测框
    for [x1, y1, x2, y2, conf] in ret:
        # 绘制矩形框
        draw.rectangle((x1, y1, x2, y2), outline='red', width=2)
    del draw
    return img


def cropDetectBoxes(img, ret):
    # 裁剪所有检测框内的图像部分
    cropped_images = []
    for [x1, y1, x2, y2, conf] in ret:
        # 裁剪图像
        cropped = img.crop((x1, y1, x2, y2))
        cropped_images.append(cropped)
    return cropped_images
