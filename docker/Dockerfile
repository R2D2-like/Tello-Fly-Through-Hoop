FROM ubuntu:20.04
SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
        wget \
        curl \
        git \
        vim \
        python3-pip \
        lsb-release \
        gnupg \
        cmake 


COPY ./. /root/Tello-Fly-Through-Hoop/
RUN chmod -R +x /root/Tello-Fly-Through-Hoop/*

# # Force color prompt in terminal.
RUN sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/' /root/.bashrc

RUN apt-get update -y && apt-get install -y libswscale-dev libavcodec-dev libavutil-dev

RUN pwd
# need for tello video
WORKDIR /root
RUN mkdir -p external/ && cd external/ && git clone https://github.com/DaWelter/h264decoder.git &&\
    cd h264decoder &&\
    mkdir build &&\
    cd build &&\
    cmake .. &&\
    cmake --build . 
WORKDIR /root

# xeyes
RUN apt-get update -y; apt-get install -y x11-apps;

# opencv
RUN pip3 install opencv-python
RUN apt-get update && \
    apt-get install -y libgl1-mesa-dev
RUN apt-get install -y libglib2.0-0

# yaml
RUN pip3 install PyYAML

# pip
RUN pip3 install --upgrade pip

RUN pip3 install djitellopy

# ping
RUN apt-get update -y && apt-get install -y iputils-ping net-tools

# tello
RUN pip3 install tello-python
RUN pip install protobuf==3.20.*

# tkinter
RUN apt-get update -y && apt-get install -y python3-tk

# opencv
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir \
    opencv-python==4.5.5.62 \
    opencv-contrib-python==4.5.5.62

# enable tab in container
RUN apt-get update && \
    apt-get install -y bash-completion && \
    rm /etc/apt/apt.conf.d/docker-clean && apt update && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*  

# Add a script to source bash-completion
RUN echo 'if [ -f /etc/bash_completion ]; then' >> /etc/bash.bashrc && \
    echo '  . /etc/bash_completion' >> /etc/bash.bashrc && \
    echo 'fi' >> /etc/bash.bashrc

WORKDIR /root

ENTRYPOINT []

CMD ["/bin/bash"]