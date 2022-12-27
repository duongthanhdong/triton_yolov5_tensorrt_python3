#set env for build TensorRT model
export ROOT_MODEL_PATH=/app/models/

export YOLOv5_MODEL=best.pt
export WTS_MODEL=yolov5n.wts
export TensorRT_MODEL=yolov5.engine
export MODEL_PATH=$ROOT_MODEL_PATH$YOLOv5_MODEL #model ultralytics/yolov5 ; put your model file in ./models file 
export MODEL_OUTPUT_TENSORRT_PATH=$ROOT_MODEL_PATH$TensorRT_MODEL
export MODEL_WTS_PATH=$ROOT_MODEL_PATH$WTS_MODEL
#modify custom object
export MODEL_TYPE=n # n,m,l,s,n6,m6,l6,s6
export MAX_OUTPUT_BBOX_COUNT=100
export CLASS_NUM=4
export INPUT_H=640
export INPUT_W=640
export	TRT_TYPE=USE_FP16 #USE_FP16 , USE_FP32
export NMS_THRESH=0.7
export CONF_THRESH=0.25
cp $MODEL_OUTPUT_TENSORRT_PATH /app
sed -i "s/INPUT_MODEL_ENGINE_PATH/$TensorRT_MODEL/" /app/server.py
sed -i "s/INPUT_LIB_libmyplugins.so/\/app\/models\/libmyplugins.so/" /app/server.py

