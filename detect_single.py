from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

conf_thres = 0.8
nms_thres = 0.4
img_size = 416

def yolov3_detect(image_path, model_def, class_path, weights_path):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    model = Darknet(model_def, img_size=img_size).to(device)

    model.load_darknet_weights(weights_path)
  

    model.eval()  # Set in evaluation mode
    
    classes = load_classes(class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


    print("\nPerforming object detection:")
    prev_time = time.time()




    #Imports and converts image for detection to pytorch tensor
    image = Image.open(image_path)

    #scales dimensions by the smaller of the width and height that scales to 416
    scaled_h = int(image.size[0] * min(img_size / image.size[0], img_size / image.size[1]))
    scaled_w = int(image.size[1] * min(img_size / image.size[0], img_size / image.size[1]))

    image = image.resize((scaled_h,scaled_w), Image.NEAREST)

    #square image of img_size by img_size to be inputted
    square_image = Image.new("RGB",(img_size,img_size))
    square_image.paste(image, ((img_size-scaled_h)//2,(img_size-scaled_w)//2))

    image = transforms.functional.to_tensor(square_image)
  


    #Alternative importing using pytorch Imagefolder
    '''
    loader = ImageFolder(image_path.rsplit("/",1)[0], img_size)

    image = None
    for loaded_images in loader:
        if image_path.endswith(str(loaded_images[0]).rsplit("\\")[1]):     
            image = loaded_images[1]
            break
    '''

    
    
    image = image.unsqueeze(0)
    input_imgs = Variable(image.type(Tensor))

    

    # Get detections
    with torch.no_grad():
        detections = model(input_imgs)
        detections = non_max_suppression(detections, conf_thres, nms_thres)
        detections = detections[0]

    # Log progress
    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)
    prev_time = current_time
    print("\t+ Batch %d, Inference Time: %s" % (0, inference_time))





    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("\nSaving image:")
    # Iterate through images and save plot of detections
    #for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):



    print("(%d) Image: '%s'" % (0, image_path))

    

    # Create plot
    img = np.array(Image.open(image_path))
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    
    # Draw bounding boxes and labels of detection
    if detections is not None:
        # Rescale boxes to original image
        

        detections = rescale_boxes(detections, img_size, img.shape[:2])
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

            print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

            box_w = x2 - x1
            box_h = y2 - y1

            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            # Create a Rectangle patch
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
            # Add the bbox to the plot
            ax.add_patch(bbox)
            # Add label
            plt.text(
                x1,
                y1,
                s=classes[int(cls_pred)],
                color="white",
                verticalalignment="top",
                bbox={"color": color, "pad": 0},
            )

    # Save generated image with detections
    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    filename = image_path.split("/")[-1].split("\\")[-1].split(".")[0]
    plt.savefig(f"output/{filename}.png",dpi=300, bbox_inches="tight", pad_inches=0.0)

    plt.show()
    plt.close()

    return detections