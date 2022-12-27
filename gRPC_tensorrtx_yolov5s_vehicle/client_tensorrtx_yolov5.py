from turbojpeg import TurboJPEG
import grpc
import detector_pb2
import detector_pb2_grpc
import cv2
import json
import io
import logging
import os
import time

def drawing_frame_custom(frame, objs):
    height_ori, width_ori = frame.shape[:2]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (128, 0, 0),
              (0, 128, 0), (0, 0, 128), (128, 0, 128), (128, 128, 0), (0, 128, 128)]
    if objs != None:
        for obj in objs:
            name = obj['name']
            class_id = obj['class_id']
            prob = obj['prob']
            x1 = int(obj['bbox'][0] )
            y1 = int(obj['bbox'][1] )
            x2 = int(obj['bbox'][2] )
            y2 = int(obj['bbox'][3] )
            cv2.rectangle(frame, (x1, y1), (x2, y2 ), colors[class_id], 2)
            str_display = str(name) + "||" + str(round(prob, 2)) + "||" + str(class_id)
            y_backgroud_classes = y1 - 35 if y1 > 30 else y1 + 35
            y_classes = y1 - 10 if y1 > 30 else y_backgroud_classes - 10
            cv2.rectangle(frame, (x1, y_backgroud_classes), (x1 + len(str_display) * 19 + 10, y1), (200,255,255), -1)
            cv2.putText(frame, name + str(round(prob, 2)), (x1 + 5, y_classes),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, colors[class_id], 3)

def drawing_frame(frame, objs):
    height_ori, width_ori = frame.shape[:2]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (128, 0, 0),
              (0, 128, 0), (0, 0, 128), (128, 0, 128), (128, 128, 0), (0, 128, 128)]
    if objs != None:
        for obj in objs:
            name = obj['name']
            class_id = obj['class_id']
            prob = obj['prob']
            x = int(obj['bbox'][0] * width_ori)
            y = int(obj['bbox'][1] * height_ori)
            w = int(obj['bbox'][2] * width_ori)
            h = int(obj['bbox'][3] * height_ori)
            cv2.rectangle(frame, (x, y), (x + w, y + h), colors[class_id], 2)

            str_display = str(name) + "||" + str(round(prob, 2)) + "||" + str(class_id)
            y_backgroud_classes = y - 35 if y > 30 else y + 35
            y_classes = y - 10 if y > 30 else y_backgroud_classes - 10
            cv2.rectangle(frame, (x, y_backgroud_classes), (x + len(str_display) * 19 + 10, y), (200,255,255), -1)
            cv2.putText(frame, name + str(round(prob, 2)), (x + 5, y_classes),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, colors[class_id], 3)

class Client_grpc_FireAndSmoke_tensorrt_yolov5():
    def __init__(self,url='192.168.1.46:30060'):
        self.channel = grpc.insecure_channel(url)
        self.stub = detector_pb2_grpc.FaceDetectorStub(self.channel)
        self.jpeg_reader = TurboJPEG()
        self.original_width = 0
        self.original_height = 0

        logging.warning("Finish Init Detection!")

    def detect(self,image,model_size= (640,640)):
        image = self.preprocess_image(image,model_size)
        image_buffer_toSend = self.jpeg_reader.encode(image)
        # _,image = cv2.imencode('.jpeg',image)
        # image_buffer_toSend = image.tobytes()
        detect = time.time()
        response = self.stub.detect(detector_pb2.Tensor(image=image_buffer_toSend))
        end_detect = time.time()
        print("Detection time =",1/(end_detect-detect))
        results = json.loads(response.objects)
        # results  = self.postprocess(results)
        return results

    def preprocess_image(self,image_raw,input_shape_model=(640,640)):
        h, w, c = image_raw.shape
        # image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        # Calculate widht and height and paddings
        r_w = input_shape_model[0] / w
        r_h = input_shape_model[1] / h
        if r_h > r_w:
            tw = input_shape_model[0]
            th = int(r_w * h)
        else:
            tw = int(r_h * w)
            th = input_shape_model[1]
        # Resize the image with long side while maintaining ratio
        image = cv2.resize(image_raw, (tw, th))
        # Pad the short side with (128,128,128)
        return image

    def postprocess(self,results,input_shape_model=(640,640)):
        r_w = input_shape_model[0]/self.original_width 
        r_h = input_shape_model[1]/self.original_height
        # print(r_w,r_h)
        for index, obj in enumerate(results):
            # print(index,obj)
            # x = int(obj['bbox'][0] * input_shape_model[0])
            # y = int(obj['bbox'][1] * input_shape_model[1])
            # w = int(obj['bbox'][2] * input_shape_model[0])
            # h = int(obj['bbox'][3] * input_shape_model[1])

            x_center = int(obj['bbox'][0])# * input_shape_model[0])
            y_center = int(obj['bbox'][1])# * input_shape_model[1])
            w = int(obj['bbox'][2])# * input_shape_model[0])
            h = int(obj['bbox'][3])# * input_shape_model[1])
            # x_center = x + w/2
            # y_center = y + h/2
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
            results[index]['bbox'] = [x1,y1,x2,y2]
        return results

def load(path="/home/gpu3080node2/Documents/vehicle",extention = "['jpg', 'jpeg', 'png']"):
    path_images = []
    for root, subdirs, files in os.walk(path):
        for name in files:
            file_ext = name.split('.')[-1]
            if file_ext.lower() in extention:
                filename = os.path.join(root, name)
                path_images.append(filename)
    return path_images


client = Client_grpc_FireAndSmoke_tensorrt_yolov5('192.168.1.63:30061')
list_data = load()
list_data = []
i = 0



for path in list_data:
    image = cv2.imread(path)
    result = client.detect(image)
    print(result)
    drawing_frame(image,result)
    # drawing_frame_custom(image,result)
    i+=1
    image_path="../output/" + str(i)+".jpg"
    print(image_path,image.shape[:2])
    cv2.imwrite(image_path,image)

import threading
import time 
image = cv2.imread("../vehicle/deae28ca-0328-11ed-a906-0d8f1b00dcd3.jpeg")
client = Client_grpc_FireAndSmoke_tensorrt_yolov5('192.168.1.63:30061')


# while 1:
#     start = time.time()
#     result = client.detect(image)
#     end = time.time()

def test(image,i):
    client = Client_grpc_FireAndSmoke_tensorrt_yolov5('10.10.10.23:30061')

    while 1:
        start = time.time()
        result = client.detect(image)
        end = time.time()
        print(i,"==",1/(end-start))

def make_many(index=100):
    for i in range(index):
        print(i)
        print(image)
        x=threading.Thread(target=test, args=(image,i))
        x.start()

make_many(5)        
