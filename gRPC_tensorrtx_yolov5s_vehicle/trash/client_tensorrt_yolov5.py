import ctypes
import os
import shutil
import random
import sys
import threading
import time
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch
import torchvision

from server_tensorrt_yolov5 import YoLov5TRT

CONF_THRESH = 0.25
IOU_THRESHOLD = 0.7
original_width = 0
original_height = 0
def preprocess_image(image_raw,input_shape_model=(640,640)):
    h, w, c = image_raw.shape
    image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
    # Calculate widht and height and paddings
    r_w = input_shape_model[0] / w
    r_h = input_shape_model[1] / h
    if r_h > r_w:
        tw = input_shape_model[0]
        th = int(r_w * h)
        tx1 = tx2 = 0
        ty1 = int((input_shape_model[1] - th) / 2)
        ty2 = input_shape_model[1] - th - ty1
    else:
        tw = int(r_h * w)
        th = input_shape_model[1]
        tx1 = int((input_shape_model[0] - tw) / 2)
        tx2 = input_shape_model[0] - tw - tx1
        ty1 = ty2 = 0
    # Resize the image with long side while maintaining ratio
    image = cv2.resize(image, (tw, th))
    # Pad the short side with (128,128,128)
    image = cv2.copyMakeBorder(
        image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
    )
    return image

def postprocess(results,input_shape_model=(640,640)):
    r_w = input_shape_model[0]/original_width 
    r_h = input_shape_model[1]/original_height
    # print(r_w,r_h)
    for index, obj in enumerate(results):


        x_center = int(obj['bbox'][0])# * input_shape_model[0])
        y_center = int(obj['bbox'][1])# * input_shape_model[1])
        w = int(obj['bbox'][2])# * input_shape_model[0])
        h = int(obj['bbox'][3])# * input_shape_model[1])

        if r_h > r_w:
            x1 = x_center - w/2
            x2 = x_center + w/2
            y1= y_center - h/2 - (input_shape_model[1] - r_w * original_height)/2
            y2= y_center + h/2 - (input_shape_model[1] - r_w * original_height)/2
            x1/=r_w
            x2/=r_w
            y1/=r_w
            y2/=r_w

        else:
            x1 = x_center - w/2 - (input_shape_model[0] - r_h * original_width) / 2
            x2 = x_center + w/2 - (input_shape_model[0] - r_h * original_width) / 2
            y1 = y_center - h/2
            y2 = y_center + h/2
            x1/=r_h
            x2/=r_h
            y1/=r_h
            y2/=r_h

        w = x2-x1
        h = y2-y1
        results[index]['bbox'] = [x1/original_width ,y1/original_height,
                                    w/original_width,h/original_height]

    return results

def bbox_to_scale_bbox(xyxy, width, height):
    x1 = int(xyxy[0] * width)
    y1 = int(xyxy[1] * height)
    x2 = int(xyxy[2] * width)
    y2 = int(xyxy[3] * height)
    if x1 < 0:
        x1 = 0
    if x2 > width:
        x2 = width
    if y1 < 0:
        y1 = 0
    if y2 > height:
        y2 = height
    return [x1, y1, x2, y2]

def drawing_frame(frame, objs):
        height_ori, width_ori = frame.shape[:2]
        if objs != None:
            for obj in objs:
                # result=self.redis_store.get_text(obj["id"])
                # print(result)

                bbox = bbox_to_scale_bbox(
                    obj["bbox"], width=width_ori, height=height_ori)
                [x, y, w, h] = bbox
                cv2.putText(frame,str(obj['prob']),(int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX ,1, (255, 0, 0), 2)
                cv2.rectangle(frame, (int(x), int(y)),
                              (int(x + w), int(y + h)), (0, 0, 255), 2)



client = YoLov5TRT('/app/tenossrt_model/yolov5.engine','/app/tenossrt_model/libmyplugins.so')
image = cv2.imread("/app/tenossrt_model/66f02d9a-16ba-11ed-a4c7-cbf20bc59202.jpeg")
ori_img = image.copy()
h, w, c = image.shape
# Calculate widht and height and paddings
input_shape_model = (640,640)
r_w = input_shape_model[0] / w
r_h = input_shape_model[1] / h
if r_h > r_w:
    tw = input_shape_model[0]
    th = int(r_w * h)
    tx1 = tx2 = 0
    ty1 = int((input_shape_model[1] - th) / 2)
    ty2 = input_shape_model[1] - th - ty1
else:
    tw = int(r_h * w)
    th = input_shape_model[1]
    tx1 = int((input_shape_model[0] - tw) / 2)
    tx2 = input_shape_model[0] - tw - tx1
    ty1 = ty2 = 0
image = cv2.resize(image, (tw, th))
original_width, original_height = tw,th
# image = preprocess_image(image,(640,640))
save = False
while True:
    
    test_image_draw = image.copy()
    # test_image_draw = cv2.cvtColor(test_image_draw, cv2.COLOR_BGR2RGB)

    input_shape_model = (640,640)
    r_w = input_shape_model[0] / w
    r_h = input_shape_model[1] / h

    if r_h > r_w:
        tw = input_shape_model[0]
        th = int(r_w * h)
        tx1 = tx2 = 0
        ty1 = int((input_shape_model[1] - th) / 2)
        ty2 = input_shape_model[1] - th - ty1
    else:
        tw = int(r_h * w)
        th = input_shape_model[1]
        tx1 = int((input_shape_model[0] - tw) / 2)
        tx2 = input_shape_model[0] - tw - tx1
        ty1 = ty2 = 0
    test_image = cv2.copyMakeBorder(
    test_image_draw, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128))

    resutl = client.infer(test_image)
    # print(resutl)
    resutl=postprocess(resutl)
    draw_image = ori_img.copy()
    drawing_frame(draw_image,resutl)
    print(resutl)
    if save is False:
        save = True
        cv2.imwrite("/app/transfer/image.jpg",draw_image)

client.destroy()