# Copyright 2015 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Python implementation of the GRPC helloworld.Greeter client."""

from __future__ import print_function
import logging
from turbojpeg import TurboJPEG
import grpc
from PIL import Image
import detector_pb2
import detector_pb2_grpc
import cv2
import time
import json
import numpy as np
import io
import pyvips
import zlib


ip = 'localhost'
port = "50051"
categories = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"]

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
            cv2.rectangle(frame, (x, y_backgroud_classes), (x + len(str_display) * 12 , y), (200,255,255), -1)
            cv2.putText(frame, name + str(round(prob, 2)), (x + 5, y_classes),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, colors[class_id], 3)

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

def pro(image):

    image = image.astype(np.float32)
    # Normalize to [0,1]
    image /= 255.0
    # HWC to CHW format:
    image = np.transpose(image, [2, 0, 1])
    # CHW to NCHW format
    image = np.expand_dims(image, axis=0)
    return image

def read_image(buffer):
    is_success, im_buf_arr = cv2.imencode(".jpeg", image)
    byte_im_jpeg = im_buf_arr.tobytes()
    while 1:
        #read from numpy
        start = time.time()
        nparr = np.frombuffer(byte_im,dtype=im_buf_arr.dtype)  
        # nparr = np.asarray(bytearray(byte_im), dtype="uint8")
        img_decode = cv2.imdecode(nparr, cv2.IMREAD_COLOR) 
        # nparr = np.frombuffer(byte_im, np.byte)
        # img2 = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
        end = time.time()


        # install pillow-simd - high performace 0.0025  >> don't install pillow 0,0047s
        image = Image.open(io.BytesIO(byte_im))#0,0047s + preprocess_image(0,0015)s = 0.0062s1
        image = np.array(image)

        ploww = time.time() -end

        # read image by TurboJPEG only jpeg image  = 0. nhanh nhat 0.022
        jpeg_reader = TurboJPEG()
        startturb = time.time()
        image = jpeg_reader.decode(byte_im_jpeg, 1)  # 1 = cv2 BGR, 0 = PIL RGB  
        endturb  = time.time()
        print(ploww,end - start,endturb - startturb)

        image = pyvips.Image.jpegload_buffer(byte_im_jpeg)
        np.ndarray(image,dtype = 'float32',shape=[640,640,3])
        print(image.shape)
            
        np_bytes = io.BytesIO()
        np.save(np_bytes, image_test, allow_pickle=True)
        np_bytes = np_bytes.getvalue()
        # load_bytes = io.BytesIO(np_bytes)
        # loaded_np = np.load(load_bytes, allow_pickle=True)

        # a = zlib.compress(bytearray(image_test))
        # print(len(a))
        # b = np.frombuffer(b,dtype='float32').resize(1,3,640,640)

        #option2
        Bytes_send = image_test.tobytes()
        # data = np.frombuffer(Bytes_send,dtype = 'float32').reshape(1,3,640,640)
        # print(np.array_equal(data, image_test))
        # print(len(Bytes_send))
        # return 0

def non_max_suppression_fast(boxes,scores, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
 
    # initialize the list of picked indexes
    pick = []
 
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    # w = boxes[:, 2]
    # h = boxes[:, 3]

    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
  
    # x2 = x1+w
    # y2 = y1+h
    # scores = boxes[:, 4]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the score of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)
 
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
 
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
                 # Calculate the upper left and lower right coordinates of the overlapping area
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        # xx2 = np.minimum(w[i]+x1[i], w[idxs[:last]]+x1[idxs[:last]])
        # yy2 = np.minimum(h[i]+y1[i], h[idxs[:last]]+y1[idxs[:last]])
        
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
                 # Calculate the length and width of the overlapping area
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
 
        # compute the ratio of overlap
                 # Calculate the area ratio of the overlapping area to the original area (overlap ratio)
        overlap = (w * h) / area[idxs[:last]]
 
                 # Delete all bounding boxes whose overlap rate is greater than the threshold
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))
 
    # return only the bounding boxes that were picked using the
    # integer data type
    return pick

def non_max_suppression(boxes, box_confidences, nms_threshold=0.5):
    x_coord = boxes[:, 0]
    y_coord = boxes[:, 1]
    width = boxes[:, 2]
    height = boxes[:, 3]
    areas = width * height
    ordered = box_confidences.argsort()[::-1]

    keep = list()
    while ordered.size > 0:
        i = ordered[0]
        keep.append(i)
        xx1 = np.maximum(x_coord[i], x_coord[ordered[1:]])
        yy1 = np.maximum(y_coord[i], y_coord[ordered[1:]])
        xx2 = np.minimum(x_coord[i] + width[i], x_coord[ordered[1:]] + width[ordered[1:]])
        yy2 = np.minimum(y_coord[i] + height[i], y_coord[ordered[1:]] + height[ordered[1:]])

        width1 = np.maximum(0.0, xx2 - xx1 + 1)
        height1 = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = width1 * height1
        union = (areas[i] + areas[ordered[1:]] - intersection)

        iou = intersection / union

        indexes = np.where(iou <= nms_threshold)[0]
        ordered = ordered[indexes + 1]
    keep = np.array(keep).astype(int)
    return keep
def post_process(output, origin_h, origin_w, conf_thresh,nms_thresh):
    results = []
    num = int(output[0])
    pred = np.reshape(output[1:],(-1,6))[:num,:]
    boxes = pred[:, :4]
    # Get the scores
    scores = pred[:, 4]
    # Get the classid
    classid = pred[:, 5]
    si = scores > conf_thresh
    boxes = boxes[si, :]
    scores = scores[si]
    classid = classid[si]
    #convert bbox x_center, y_center ,w,h => xmin,ymin,w,h
    boxes[:,0] = boxes[:,0] - boxes[:,2]/2
    boxes[:,1] = boxes[:,1] - boxes[:,3]/2
    #do nmx
    indices = non_max_suppression(boxes,scores,nms_thresh)

    result_boxes = boxes[indices, :]
    result_scores = scores[indices]
    result_classid = classid[indices]
    for j in range(len(result_boxes)):
        temp = {}
        x1 = int(result_boxes[j][0])/origin_w
        y1 = int(result_boxes[j][1])/origin_h
        w = int(result_boxes[j][2])/origin_w
        h = int(result_boxes[j][3])/origin_h
        
        class_id = int(result_classid[j])
        name = categories[class_id]
        prob = result_scores[j]
        temp = {
            "name":str(name),
            "class_id":class_id,
            "prob": float(prob),
            'bbox': [x1,y1,w,h]
        }   
        results.append(temp)
    return results

def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    options = [('grpc.max_message_length', 100 * 1024 * 1024)]
    with grpc.insecure_channel('192.168.1.47:30058',options=options) as channel:
        stub = detector_pb2_grpc.FaceDetectorStub(channel)

        image = cv2.imread('bus.jpg')
        image = preprocess_image(image,(640,640))
        cv2.imwrite('imageaa1.jpg',image)

        print(image.dtype)
        # image_test = pro(image)
        # image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        # image = cv2.resize(image, (112, 112))
        start = time.time()
        is_success, im_buf_arr = cv2.imencode(".jpeg", image)
        byte_im = im_buf_arr.tobytes()
        end = time.time()
        print(end -start)
        import pyvips
        # image = pyvips.Image.new_from_memory(image,640,640,3,"uchar")
        jpeg_reader = TurboJPEG() ## 0.002
        start = time.time()
        image_bufer = jpeg_reader.encode(image)
        end = time.time()
        print(end-start)
        image = jpeg_reader.decode(image_bufer)
        cv2.imwrite('imageaa.jpg',image)
        # return
        # files = {'body': byte_im}
        while 1:
            start = time.time()
            response = stub.detect(detector_pb2.Tensor(image=image_bufer))
            end = time.time()
            elapsed = end - start
            fps = 1 / elapsed
            print(elapsed, fps)
            detection = response.objects
            detection = json.loads(detection)
            start = time.time()
            json.dumps(detection)
            dump_json=time.time() - start

            start = time.time()
            # results = post_process(detection,640,640,0.25,0.7)
            post_process_time =  (time.time() - start)
            start = time.time()
            # json.dumps(results)
            dump_json1 = (time.time() - start)
            print("Dum:{} - (post{}+dump{})  = {}".format(dump_json,post_process_time,dump_json1,dump_json-(post_process_time+dump_json1)))
            break
            # drawing_frame(image, detection)

        cv2.imwrite("image_grpc_test.jpg",image)



# print(response)

if __name__ == '__main__':
    logging.basicConfig()
    run()
