FROM nvcr.io/nvidia/tensorrt:20.12-py3
ENV APP_DIR /app
ENV PYTHONUNBUFFERED TRUE
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y wget python3-dev gcc && \
    apt-get install -y --no-install-recommends libopencv-dev && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py
    
WORKDIR ${APP_DIR}
COPY requirements.txt ${APP_DIR}
RUN pip3 install -r requirements.txt

RUN pip3 install opencv-python
RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install pycuda

RUN apt-get update && apt-get install libturbojpeg
RUN pip3 install PyTurboJPEG


RUN apt-get update && \
    apt-get install libb64-dev libb64-0d


WORKDIR ${APP_DIR}

# if you want compile it in manual, just ignore this one 
# BUILD model TensorRT *.engine 

RUN git clone https://github.com/ultralytics/yolov5.git
RUN chmod 777 -R yolov5

RUN git clone https://github.com/wang-xinyu/tensorrtx.git
RUN chmod 777 -R tensorrtx

RUN cp ./tensorrtx/yolov5/gen_wts.py ./yolov5
COPY requirements_for_buildmodel.txt /app
RUN pip3 install -r requirements_for_buildmodel.txt

# End of set env build TensorRT 


COPY . ${APP_DIR}


# RUN BUILD 
# remember edit ENV in create_model_tensorrt.sh before RUN IT 

COPY models/create_model_tensorrt.sh .
RUN bash create_model_tensorrt.sh
#ENTRYPOINT ["bash","create_model_tensorrt.sh"]
#ENTRYPOINT bash create_model_tensorrt.sh

#Run it when you have *.engine 
#COPY models/modify_sever.sh .
#RUN bash modify_sever.sh
#ENTRYPOINT python3 tenossrt_model/client_tensorrt_yolov5.py
