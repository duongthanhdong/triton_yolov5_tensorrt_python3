# 
# Docker will download the TensorRT container. You need to select the version (in this case 20.08) according to the version of Triton that you want to use later to ensure the TensorRT versions match. Matching NGC version tags use the same TensorRT version.
###

FROM nvcr.io/nvidia/tensorrt:20.12-py3

ARG DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

RUN mkdir -p /workspace

# Install requried libraries
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev \
    wget \
    zlib1g-dev \
    git \
    pkg-config \
    sudo \
    ssh \
    libssl-dev \
    pbzip2 \
    pv \
    bzip2 \
    unzip \
    devscripts \
    lintian \
    fakeroot \
    dh-make \
    build-essential

# Install python3
RUN apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      python3-dev \
      python3-wheel &&\
    cd /usr/local/bin &&\
    ln -sf /usr/bin/python3 python &&\
    ln -sf /usr/bin/pip3 pip;
    
# Install PyPI packages
RUN pip3 install --upgrade pip
RUN pip3 install setuptools>=41.0.0
COPY requirements.txt /tmp/requirements.txt
#RUN pip3 install -r /tmp/requirements.txt

# For opencv
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /workspace

RUN apt -y install build-essential cmake git pkg-config libgtk-3-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
    gfortran openexr libatlas-base-dev python3-dev python3-numpy \
    libtbb2 libtbb-dev libdc1394-22-dev nano
    
RUN wget https://github.com/opencv/opencv/archive/4.5.5.zip -O opencv.zip \
    && unzip opencv.zip 
RUN wget https://github.com/opencv/opencv_contrib/archive/refs/tags/4.5.5.zip -O opencv_contrib.zip \
    && unzip opencv_contrib 

RUN mkdir /workspace/opencv-4.5.5/cmake_binary 
     
RUN cd /workspace/opencv-4.5.5/cmake_binary \
    && cmake -DOPENCV_EXTRA_MODULES_PATH=/workspace/opencv_contrib-4.5.5/modules \
      -DBUILD_TIFF=ON \
      -DBUILD_opencv_java=OFF \
      -DWITH_CUDA=OFF \
      -DENABLE_AVX=ON \
      -DWITH_OPENGL=ON \
      -DWITH_OPENCL=ON \
      -DWITH_IPP=ON \
      -DWITH_TBB=ON \
      -DWITH_EIGEN=ON \
      -DWITH_V4L=ON \
      -DBUILD_TESTS=OFF \
      -DBUILD_PERF_TESTS=OFF \
      -DCMAKE_BUILD_TYPE=RELEASE \
      -DBUILD_opencv_python3=ON \
      -DCMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
      -DPYTHON_EXECUTABLE=$(which python3) \
      -DPYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
      -DPYTHON_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") .. 

RUN cd /workspace/opencv-4.5.5/cmake_binary \
    && make install -j16 
RUN rm /workspace/opencv.zip \
    && rm /workspace/opencv_contrib.zip \
    && rm -r /workspace/opencv-4.5.5 \
    && rm -r /workspace/opencv_contrib-4.5.5

RUN cd /usr/local/lib/python3.8/dist-packages/ \
    && ln -s /usr/lib/python3.8/site-packages/cv2/python-3.8/cv2.cpython-38-x86_64-linux-gnu.so cv2.so

RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install pycuda
RUN mkdir tensorrtx
COPY tensorrtx tensorrtx 
COPY yolov5s.pt /workspace
RUN pip3 install -r /tmp/requirements.txt

COPY convert.sh ./convert.sh
RUN chmod +x ./convert.sh
RUN bash convert.sh
# RUN ["/bin/bash"]
#ENTRYPOINT ["/bin/bash", "./convert.sh"]