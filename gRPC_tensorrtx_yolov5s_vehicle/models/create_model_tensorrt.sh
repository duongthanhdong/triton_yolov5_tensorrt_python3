#set env for build TensorRT model

#dont edit 
export ROOT_MODEL_PATH=/app/models/
export TensorRT_MODEL=yolov5.engine

# edit 
export YOLOv5_MODEL=yolov5s-vehicle.pt
export WTS_MODEL=yolov5.wts
export MODEL_PATH=$ROOT_MODEL_PATH$YOLOv5_MODEL #model ultralytics/yolov5 ; put your model file in ./models file 
export MODEL_OUTPUT_TENSORRT_PATH=$ROOT_MODEL_PATH$TensorRT_MODEL
export MODEL_WTS_PATH=$ROOT_MODEL_PATH$WTS_MODEL
#modify custom object
export MODEL_TYPE=s # n,s,m,l,n6,m6,l6,s6
export MAX_OUTPUT_BBOX_COUNT=500
export CLASS_NUM=4
export INPUT_H=640
export INPUT_W=640
export	TRT_TYPE=USE_FP16 #USE_FP16 , USE_FP32
export NMS_THRESH=0.7
export CONF_THRESH=0.25


cd /app/yolov5

#genegate WTS model
python3 gen_wts.py -w ${MODEL_PATH} -o $MODEL_WTS_PATH

echo "Wts is genarate in ${MODEL_WTS_PATH}"

# modify custom object
cd /app/tensorrtx/yolov5/
sed -i "s/static constexpr int MAX_OUTPUT_BBOX_COUNT = 1000;/static constexpr int MAX_OUTPUT_BBOX_COUNT = ${MAX_OUTPUT_BBOX_COUNT};/" yololayer.h
sed -i "s/static constexpr int CLASS_NUM = 80;/static constexpr int CLASS_NUM = ${CLASS_NUM};/" yololayer.h
sed -i "s/static constexpr int INPUT_H = 640;/static constexpr int INPUT_H = ${INPUT_H};/" yololayer.h
sed -i "s/static constexpr int INPUT_W = 640;/static constexpr int INPUT_W = ${INPUT_W};/" yololayer.h

sed -i "s/#define USE_FP16/#define ${TRT_TYPE}/" yolov5.cpp
sed -i "s/#define NMS_THRESH 0.4/#define NMS_THRESH ${NMS_THRESH}/" yolov5.cpp
sed -i "s/#define CONF_THRESH 0.5/#define CONF_THRESH ${CONF_THRESH}/" yolov5.cpp

#build  ./yolov5  build tools
mkdir /app/tensorrtx/yolov5/build/
cd /app/tensorrtx/yolov5/build/
cmake .. && make


#build model TensorRT engine
./yolov5 -s $MODEL_WTS_PATH $MODEL_OUTPUT_TENSORRT_PATH $MODEL_TYPE
echo "Finish create process . File was saved in $MODEL_OUTPUT_TENSORRT_PATH"


#copy file to local app dir for run sever.py

cp libmyplugins.so $ROOT_MODEL_PATH 
cp $MODEL_OUTPUT_TENSORRT_PATH /app

#modify path of model and lib 
export APP_DIR=/app/
sed -i "s/INPUT_MODEL_ENGINE_PATH/$TensorRT_MODEL/" /app/server.py
sed -i "s/INPUT_LIB_libmyplugins.so/\/app\/models\/libmyplugins.so/" /app/server.py

 cd /app
 #python3 server.py -p 30061

