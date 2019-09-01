FROM ubuntu

RUN apt-get update && apt-get install -y \
    libgsl-dev \
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
    git \
    make \
    build-essential \
    liblapacke-dev \
    vim

RUN cd home &&\ 
    git clone https://github.com/DReichLab/EIG.git &&\
    cd EIG/src &&\
    make LDLIBS="-llapacke" &&\
    make install
