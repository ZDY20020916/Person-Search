from ultralytics import YOLO
from yolov8.utils.yolov8utils import _xywh_to_xyxy, classes_names_coco, drawDetectBox, cropDetectBoxes
from utils.imgprocess import read_image


def detect_original(img_path, conf_thresh=0.5):
    model = YOLO('yolov8n.pt')
    results = model(img_path)
    ret = []
    img = read_image(img_path)
    width, height = img.size
    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = round(float(box.conf[0]), 2)
            cls = int(box.cls[0])  # 获取物体类别标签
            if conf < conf_thresh or cls != 0:
                continue
            x1, y1, x2, y2 = map(int, _xywh_to_xyxy(box.xywh[0], width, height))
            class_name = classes_names_coco[cls]
            ret.append([x1, y1, x2, y2, conf])
        break
    processed_img = drawDetectBox(img.copy(), ret)
    croppedboxes = cropDetectBoxes(img.copy(), ret)
    # for i, croppedbox in enumerate(croppedboxes):
    #     croppedbox.save('bus'+str(i)+'.jpg')
    return ret, processed_img, croppedboxes


if __name__ == "__main__":
    detect_original("bus.jpg")