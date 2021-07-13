from yolov3 import yolov3
from PIL import Image
import pyautogui as pag
import time

cfg = "config/yolov3-tiny-rs.cfg"
weights = "weights/yolov3-tiny-rs.weights"
classes = "data/RS/rs.names"

image_path = "data/RS/images/morecows6.jpg"
image = Image.open(image_path)
#image_path = "screen"




print("Detecting in " + image_path)

#yolov3_detect(image_path, cfg, classes, weights)
yolo = yolov3(cfg, classes, weights, nms_thres=0.8)


#screen = pag.screenshot()
print("Performing detection...", end="\r")
output = yolo(image)

if output:
    print("Detection complete. Number of detections: " + str(len(output)), end = "\r")
else:
    print("No detections")


time.sleep(1)