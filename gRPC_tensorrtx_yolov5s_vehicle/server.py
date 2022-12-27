# Copyright 2015 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Python implementation of the GRPC helloworld.Greeter server."""

from concurrent import futures
import logging

import grpc
import argparse
import detector_pb2
import detector_pb2_grpc
import time
from PIL import Image
from tenossrt_model.server_tensorrt_yolov5 import YoLov5TRT
import cv2
import io
import numpy as np
import json
from turbojpeg import TurboJPEG

def preprocess_image(image_raw,input_shape_model=(640,640)):
    h, w, c = image_raw.shape
    image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
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
    image = cv2.resize(image_raw, (tw, th))
    # Pad the short side with (128,128,128)
    image = cv2.copyMakeBorder(
        image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
    )
    return image

class ServerYoloV5(detector_pb2_grpc.FaceDetectorServicer):

    def __init__(self):
        self.__yolo = YoLov5TRT("INPUT_MODEL_ENGINE_PATH","INPUT_LIB_libmyplugins.so")
        # self.__yolo = YoLov5TRT("models/yolov5n.engine","models/libmyplugins.so")

        self.__jpeg_reader = TurboJPEG()

    def detect(self, request, context):
        start = time.time()
        image = self.__jpeg_reader.decode(request.image)  # 1 = cv2 BGR, 0 = PIL RGB  
        image = self.preprocess_image(image)
        detection = self.__yolo.infer(img= image, conf_thresh=.25,nms_thresh=.7)
        detection = self.postprocess(detection)
        # end_detection = time.time()-end
        # detection[2] = detection[2].tolist()
        detection = json.dumps(detection)
        end_function = time.time() - start
        print("fuction FPS= ",1/end_function)
        return detector_pb2.ObjectInfo(objects='%s' % detection)

    def preprocess_image(self,image_raw,input_shape_model=(640,640)):
        h, w, c = image_raw.shape
        self.original_width = w
        self.original_height = h
        image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
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
        # image = cv2.resize(image_raw, (tw, th))
        # Pad the short side with (128,128,128)
        image = cv2.copyMakeBorder(
            image_raw, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
        )
        return image

    def postprocess(self,results,input_shape_model=(640,640)):
        r_w = input_shape_model[0]/self.original_width 
        r_h = input_shape_model[1]/self.original_height
        # print(r_w,r_h)
        for index, obj in enumerate(results):
    

            x_center = int(obj['bbox'][0])# * input_shape_model[0])
            y_center = int(obj['bbox'][1])# * input_shape_model[1])
            w = int(obj['bbox'][2])# * input_shape_model[0])
            h = int(obj['bbox'][3])# * input_shape_model[1])

            if r_h > r_w:
                x1 = x_center - w/2
                x2 = x_center + w/2
                y1= y_center - h/2 - (input_shape_model[1] - r_w * self.original_height)/2
                y2= y_center + h/2 - (input_shape_model[1] - r_w * self.original_height)/2
                x1/=r_w
                x2/=r_w
                y1/=r_w
                y2/=r_w

            else:
                x1 = x_center - w/2 - (input_shape_model[0] - r_h * self.original_width) / 2
                x2 = x_center + w/2 - (input_shape_model[0] - r_h * self.original_width) / 2
                y1 = y_center - h/2
                y2 = y_center + h/2
                x1/=r_h
                x2/=r_h
                y1/=r_h
                y2/=r_h

            w = x2-x1
            h = y2-y1
            results[index]['bbox'] = [x1/self.original_width ,y1/self.original_height,
                                        w/self.original_width,h/self.original_height]
        return results

def serve(port,max_workers):
    options = [('grpc.max_message_length', 100 * 1024 * 1024)]

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    detector_pb2_grpc.add_FaceDetectorServicer_to_server(ServerYoloV5(), server)
    server.add_insecure_port('[::]:'+port)
    print("start server in ",port,"| max_workers=",max_workers)

    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='add enviroment')
    # parser.add_argument('--video_path', default='./video/video3.mp4',
    #                     help='path to your input video (defaulte is "VMS.mp4")')
    parser.add_argument('--port','-p', default='50051',
                        help='input the port for service')
    parser.add_argument('--workers','-w', default=1,
                        help='input the port for service')
    args = parser.parse_args()
    logging.basicConfig()
    serve(args.port,args.workers)
