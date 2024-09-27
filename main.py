#%%
import os
import sys
import json
import numpy as np
import time
from PIL import Image, ImageDraw

ROOT_DIR = 'Mask-RCNN_TF2.14.0'

sys.path.append(ROOT_DIR)
from mrcnn.config import Config
import mrcnn.utils as utils
#from mrcnn import visualize
import mrcnn.model as modellib

#%%
class_names = ["BG", "Parking Space", "Drivable Space"]
# class_names = ["BG", "Car", "Van", "Other Vehicle", "Motorbike", "Bicycle", "Electric Scooter", "Adult", "Child", "Stroller", "Shopping Cart", "Gate Arm",
#                "Parking Block", "Speed Bump", "Traffic Pole", "Traffic Cone", "Traffic Drum", "Traffic Barricade", "Cylindrical Bollard", "U-shaped Bollard",
#                "Other Road Barriers", "No Parking Stand", "Adjustable Parking Pole", "Waste Tire", "Planter Barrier", "Water Container", "Movable Obstacle",
#                "Barrier Gate", "Electric Car Charger", "Parking Meter", "Parking Sign", "Traffic Light", "Pedestrian Light", "Street Sign", "Disabled Parking Space",
#                "Pregnant Parking Space", "Electric Car Parking Space", "Two-wheeled Vehicle Parking Space", "Other Parking Space"]
#%%

class InferenceConfig(Config):
    NAME = "bbox"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = len(class_names)
    DETECTION_MIN_CONFIDENCE = 0.9

inference_config = InferenceConfig()

model_path = "mask_rcnn_seg_0100.h5"
# model_path = "/content/mask_rcnn_bbox_0100.h5"

test_model = modellib.MaskRCNN(
    mode="inference",
    config=inference_config,
    model_dir=model_path)

#test_model.keras_model.save_weights('/content/mask_rcnn_seg_0100_new.tf')

test_model.load_weights(model_path, by_name=True)
# %%
import skimage

# %%
import cv2
from tqdm import tqdm

mask_colors_255 = [
    (0, 0, 0), # Background
    (0, 255, 0), # Parking space
    (0, 0, 255)  # Drivable space
]

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('output.mp4', fourcc, 10, (1920, 1080))

current_path = os.getcwd()

test_item = 'parking-space-indoor/대형주차장_002/Camera'

real_test_dir = os.path.join(current_path, test_item)
image_paths = []

for filename in sorted(os.listdir(real_test_dir)):
    if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
        image_paths.append(os.path.join(real_test_dir, filename))

for image_path in tqdm(image_paths):
    img = skimage.io.imread(image_path)
    img_arr = np.array(img)

    results = test_model.detect([img_arr])

    rois = results[0]['rois']
    class_ids = results[0]['class_ids']
    scores = results[0]['scores']
    masks = results[0]['masks']

    result_img = img.copy()

    for i, class_id in enumerate(class_ids):
        mask = masks[:, :, i].astype(np.float32)
        mask = (mask * 255).astype(np.uint8)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result_img, contours, 0, mask_colors_255[class_id], 2)

        x, y, w, h = cv2.boundingRect(contours[0])
        # cv2.rectangle(result_img, (x, y), (x + w, y + h), (255, 255, 255), 2)

    out.write(result_img)

out.release()


