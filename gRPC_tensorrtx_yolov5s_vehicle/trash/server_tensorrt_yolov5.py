import ctypes
import os
import shutil
import random
import sys
import threading
import time
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

CONF_THRESH = 0.25
IOU_THRESHOLD = 0.7

class YoLov5TRT(object):
    """
    description: A YOLOv5 class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_file_path,lib_so_path):
        # Create a Context on this device,
        ctypes.CDLL(lib_so_path)
        self.ctx = cuda.Device(0).make_context()
        self.ctx.push()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            print('bingding:', binding, engine.get_binding_shape(binding))
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = engine.max_batch_size
        self.ctx.pop()
        print("Finish Init Tensorrt object")


    def infer(self, img, conf_thresh = 0.25, nms_thresh=0.7 ):
        # start = time.time()
        # Make self the active context, pushing it on top of the context stack.
        self.ctx.push()
        # Restore
        stream = self.stream
        context = self.context
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        # end_init = time.time()
        # Do image preprocess
        input_image= self.preprocess_image(img)#
        # end_preprocess = time.time()
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], input_image.ravel(), stream)
        # Run inference.
        context.execute_async(batch_size=self.batch_size, bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        # Synchronize the stream
        stream.synchronize()
        # end_detect = time.time()
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        # print("init= {} | preprocess= {} | detect= {} | pop= {}".format(end_init-start,end_preprocess-end_init,end_detect-end_preprocess,end_pop - end_detect))
        results = self.postprocess( host_outputs[0], 640, 640, conf_thresh,nms_thresh)
        # Do postprocess 0,00095
        # results = []
        # print(host_outputs[0])  
        # # print(type(host_outputs[0]))
        # result_boxes, result_scores, result_classid = self.post_process(
        #     host_outputs[0], origin_h, origin_w,conf_thresh,nms_thresh
        # )
        # post_process = time.time()

        # # Draw rectangles and labels on the original image
        # for j in range(len(result_boxes)):
        #     temp = {}
        #     x1 = int(result_boxes[j][0])/origin_w
        #     y1 = int(result_boxes[j][1])/origin_h
        #     x2 = int(result_boxes[j][2])/origin_w
        #     y2 = int(result_boxes[j][3])/origin_h
        #     w = x2 - x1
        #     h = y2 - y1
        #     class_id = int(result_classid[j])
        #     name = categories[class_id]
        #     prob = result_scores[j]
        #     temp = {
        #         "name":str(name),
        #         "class_id":class_id,
        #         "prob": float(prob),
        #         'bbox': [x1,y1,w,h]
        #     }   
        #     results.append(temp)
        # time_post_process = time.time() - post_process

        return results

    def preprocess_image(self, raw_bgr_image):
        """
        description: Convert BGR image to RGB,
                     resize and pad it to target size, normalize to [0,1],
                     transform to NCHW format.
        param:
            input_image_path: str, image path
        return:
            image:  the processed image
            image_raw: the original image
            h: original height
            w: original width
        """
        image = raw_bgr_image.astype(np.float32)
        # Normalize to [0,1]
        image /= 255.0
        # HWC to CHW format:
        import time
        start =time.time()
        image = np.transpose(image, [2, 0, 1])[::-1]   # HWC to CHW, BGR to RGB
        end = time.time()
        image123 = np.transpose(image, [2, 0, 1])
        end_2  = time.time()

        print( 1/(end-start),1/(end_2-end))
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)
        return image

    def non_max_suppression(self,boxes, box_confidences, nms_threshold=0.5):
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
            #x1, y1, w, h
            # xx1 = np.maximum(x_coord[i], x_coord[ordered[1:]])
            # yy1 = np.maximum(y_coord[i], y_coord[ordered[1:]])
            # xx2 = np.minimum(x_coord[i] + width[i], x_coord[ordered[1:]] + width[ordered[1:]])
            # yy2 = np.minimum(y_coord[i] + height[i], y_coord[ordered[1:]] + height[ordered[1:]])

            #custome for x_center,y_center, w,h
            xx1 = np.maximum(x_coord[i] - width[i]/2, x_coord[ordered[1:]] - width[ordered[1:]]/2)
            yy1 = np.maximum(y_coord[i] - height[i]/2, y_coord[ordered[1:]] - height[ordered[1:]]/2)
            xx2 = np.minimum(x_coord[i] + width[i]/2, x_coord[ordered[1:]] + width[ordered[1:]]/2 )
            yy2 = np.minimum(y_coord[i] + height[i]/2, y_coord[ordered[1:]] + height[ordered[1:]]/2 )

            width1 = np.maximum(0.0, xx2 - xx1 + 1)
            height1 = np.maximum(0.0, yy2 - yy1 + 1)
            intersection = width1 * height1
            union = (areas[i] + areas[ordered[1:]] - intersection)

            iou = intersection / union

            indexes = np.where(iou <= nms_threshold)[0]
            ordered = ordered[indexes + 1]
        keep = np.array(keep).astype(int)
        return keep

    def postprocess(self,output, origin_h, origin_w, conf_thresh,nms_thresh):
        results = []
        num = int(output[0])
        pred = np.reshape(output[1:],(-1,6))[:num,:]
        boxes = pred[:, :4]
        # Get the scores
        scores = pred[:, 4]
        # Get the classid
        classid = pred[:, 5]
        si = scores > conf_thresh
        # c = np.logical_and(scores > conf_thresh, classid == 1)
        boxes = boxes[si, :]
        scores = scores[si]
        classid = classid[si]
        
        # sa = classid == 1

        # boxes = boxes[sa, :]
        # scores = scores[sa]
        # classid = classid[sa]
        
        #convert bbox x_center, y_center ,w,h => xmin,ymin,w,h

        # boxes[:,0] = boxes[:,0] - boxes[:,2]/2
        # boxes[:,1] = boxes[:,1] - boxes[:,3]/2

        #do nmx
        indices = self.non_max_suppression(boxes,scores,nms_thresh)
        
        # boxes[:,0] = boxes[:,0] + boxes[:,2]/2
        # boxes[:,1] = boxes[:,1] + boxes[:,3]/2
        
        result_boxes = boxes[indices, :]
        result_scores = scores[indices]
        result_classid = classid[indices]
        for j in range(len(result_boxes)):
            x_center = int(result_boxes[j][0])#/origin_w
            y_center = int(result_boxes[j][1])#/origin_h
            w = int(result_boxes[j][2])#/origin_w
            h = int(result_boxes[j][3])#/origin_h
            class_id = int(result_classid[j])
            results.append({
                "name":str(categories[class_id]),
                "class_id":class_id,
                "prob": float(result_scores[j]),
                'bbox': [x_center,y_center,w,h]
            })
        return results

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()

categories = ["truck",'bus','car','motobike']

# print("load model")
# client = YoLov5TRT('build/yolov5m_FP16.engine','build/libmyplugins.so')
# print("detect start")
# def test(index):
#     i=0

#     while 1:
#         image = cv2.imread("samples/bus.jpg")
#         image = preprocess_image1(image)

#         results ,FPS = client.infer([image])

#         # print(results)
#         # print(FPS)
#         print("FPS {}: ".format(str(index)) ,str(1/FPS))
#         i+=1
#         if i > 1000:
#             client.destroy()
#             break
#     print("detect ending")
# def make_many(index=100):
#     import threading
#     for i in range(index):
#         x=threading.Thread(target=test, args=(i,))
#         x.start()
# make_many(20)