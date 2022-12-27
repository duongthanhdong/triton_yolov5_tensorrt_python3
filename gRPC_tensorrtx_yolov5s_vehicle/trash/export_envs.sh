export MODEL_WTS_NAME=best.wts
export MODEL_TYPE=m
export MAX_OUTPUT_BBOX_COUNT=100
export CLASS_NUM=2
export INPUT_H=640
export INPUT_W=640
export	TRT_TYPE=USE_FP16
export NMS_THRESH=0.7
export CONF_THRESH=0.25

ENV MODEL_WTS_NAME best.wts
ENV MODEL_TYPE m
ENV MAX_OUTPUT_BBOX_COUNT 100
ENV CLASS_NUM 2
ENV INPUT_H 640
ENV INPUT_W 640
ENV	TRT_TYPE USE_FP16
ENV NMS_THRESH 0.7
ENV CONF_THRESH 0.25

RUN sed -i "s/static constexpr int MAX_OUTPUT_BBOX_COUNT = 1000;/static constexpr int MAX_OUTPUT_BBOX_COUNT = ${MAX_OUTPUT_BBOX_COUNT};/" yololayer.h
RUN sed -i "s/static constexpr int CLASS_NUM = 80;/static constexpr int CLASS_NUM = ${CLASS_NUM};/" yololayer.h
RUN sed -i "s/static constexpr int INPUT_H = 640;/static constexpr int INPUT_H = ${INPUT_H};/" yololayer.h
RUN sed -i "s/static constexpr int INPUT_W = 640;/static constexpr int INPUT_W = ${INPUT_W};/" yololayer.h

RUN sed -i "s/#define USE_FP16/#define ${TRT_TYPE}/" yolov5.cpp
RUN sed -i "s/#define NMS_THRESH 0.4/#define NMS_THRESH ${NMS_THRESH}/" yolov5.cpp
RUN sed -i "s/#define CONF_THRESH 0.5/#define CONF_THRESH ${CONF_THRESH}/" yolov5.cpp
RUN cmake .. && make