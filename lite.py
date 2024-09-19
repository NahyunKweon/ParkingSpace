# %%
import tensorflow as tf

# Eager execution 활성화
tf.compat.v1.enable_eager_execution()
print("Eager execution:", tf.executing_eagerly())

import os
import sys
import json
import numpy as np
print("Eager execution:", tf.executing_eagerly())
import time
from PIL import Image, ImageDraw

ROOT_DIR = 'Mask_RCNN'

sys.path.append(ROOT_DIR)
from mrcnn.config import Config
import mrcnn.utils as utils
from mrcnn import visualize
import mrcnn.model as modellib
print("Eager execution:", tf.executing_eagerly())
#%%
class_names = ["BG", "Parking Space", "Drivable Space"]
# class_names = ["BG", "Car", "Van", "Other Vehicle", "Motorbike", "Bicycle", "Electric Scooter", "Adult", "Child", "Stroller", "Shopping Cart", "Gate Arm",
#                "Parking Block", "Speed Bump", "Traffic Pole", "Traffic Cone", "Traffic Drum", "Traffic Barricade", "Cylindrical Bollard", "U-shaped Bollard",
#                "Other Road Barriers", "No Parking Stand", "Adjustable Parking Pole", "Waste Tire", "Planter Barrier", "Water Container", "Movable Obstacle",
#                "Barrier Gate", "Electric Car Charger", "Parking Meter", "Parking Sign", "Traffic Light", "Pedestrian Light", "Street Sign", "Disabled Parking Space",
#                "Pregnant Parking Space", "Electric Car Parking Space", "Two-wheeled Vehicle Parking Space", "Other Parking Space"]

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

test_model.load_weights(model_path, by_name=True)


#%%
print("Eager execution:", tf.executing_eagerly())
# TFLite Converter 생성
converter = tf.lite.TFLiteConverter.from_keras_model(test_model.keras_model)

# Set the target ops to TensorFlow Lite Ops

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter._experimental_lower_tensor_list_ops = True
# Optional: Adjust conversion options
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 모델 변환
# Initialize tflite_model
tflite_model = None

# Try to convert the model
try:
    tflite_model = converter.convert()
except Exception as e:
    print("Error during conversion:", e)

# Save the TFLite model if conversion was successful
if tflite_model is not None:
    with open('mask_rcnn_model.tflite', 'wb') as f:
        f.write(tflite_model)
else:
    print("Model conversion failed; TFLite model not created.")
# %%
import tensorflow as tf

# TFLite 모델 파일 경로
tflite_model_path = 'mask_rcnn_model.tflite'

# TFLite 모델 로드
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)

# 모델 초기화
interpreter.allocate_tensors()

# 입력 및 출력 텐서 인덱스 얻기
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 입력 텐서 정보 출력
print("Input details:", input_details)
print("Output details:", output_details)
# %%
