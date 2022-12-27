#!/bin/bash
cd /workspace
mkdir yolov5
echo "1"
cp yolov5s.pt yolov5
echo "2"
cp tensorrtx/yolov5/gen_wts.py yolov5

cd /workspace/yolov5
echo "3"
# pip install -r requirements.txt

python gen_wts.py -w /workspace/yolov5/yolov5s.pt -o /workspace/yolov5/yolov5s.wts
echo "4"
cd /workspace/tensorrtx/yolov5
echo "5"
rm -rf /workspace/tensorrtx/yolov5/build

mkdir -p /workspace/tensorrtx/yolov5/build

cd /workspace/tensorrtx/yolov5/build
# update CLASS_NUM in yololayer.h if your model is trained on custom dataset
cp /workspace/yolov5/yolov5s.wts /workspace/tensorrtx/yolov5/build
cmake ..
make -j8

/workspace/tensorrtx/yolov5/build/yolov5 -s /workspace/tensorrtx/yolov5/build/yolov5s.wts /workspace/tensorrtx/yolov5/build/yolov5s.engine m6

exec "$@"
