# ARG UBUNTU_VER=22.04
# FROM ubuntu:${UBUNTU_VER}

FROM ros:humble

# Use bash shell
SHELL ["/bin/bash", "--login", "-c"]

# Set environment variables to non-interactive (this prevents some prompts)
ENV DEBIAN_FRONTEND=non-interactive
ENV PATH="/usr/local/bin:${PATH}"

# System packages 
RUN apt-get update -y \
    && apt-get upgrade -y

RUN apt -y purge binutils \
    && apt -y remove make \
    && apt -y autoremove 
    
RUN apt-get install -y vim \
    net-tools \
    build-essential \
    git \
    liblapack-dev \
    libopenblas-dev \
    libeigen3-dev \
    libboost-all-dev \
    libglib2.0-dev \
    libpoco-dev \
    curl \
    ca-certificates \
    python3.11 \
    python3.11-distutils \
    python3.11-dev \
    python3.11-venv

# Set python3.11 as default python3 and python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --set python3 /usr/bin/python3.11 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && ln -sf /usr/local/bin/pip3.11 /usr/local/bin/pip

# Install minimal LaTeX packages for English
RUN apt-get install -y --no-install-recommends \
    texlive-latex-recommended \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-base \
    cm-super \
    dvipng \
    && texhash

# Display
RUN apt-get install -y -q --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

# Install the required Python packages
# RUN pip3 install numpy \
#     scipy==1.11.4 \
#     matplotlib \
#     proxsuite \
#     pin \
#     mujoco \
#     osqp \
#     ipykernel \
#     empy==3.3.4 \
#     catkin_pkg \
#     lark \
#     PyQt6 \
#     "pybind11[global]"

RUN apt-get remove -y python3-psutil

RUN pip3 install matplotlib \
    ipykernel \
    PyQt5 \
    PySide2

# Install uaibot
RUN git clone --recursive https://github.com/shiqingw/UAIbotPy.git \
    && cd UAIbotPy \
    && pip install .

# Install libfranka
RUN git clone --recursive https://github.com/frankaemika/libfranka --branch 0.13.3 \
    && cd libfranka \
    && mkdir build \
    && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF .. \
    && cmake --build . \
    && cpack -G DEB \
    && dpkg -i libfranka*.deb \
    && cd / \
    && rm -rf libfranka

# Install LCM (checkout at an older version)
RUN git clone https://github.com/lcm-proj/lcm.git \
    && cd lcm \
    && git checkout 6f2f4c948667452aea021c0afefbd512f91283bc \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make \
    && make install \
    && cd /lcm/lcm-python \
    && pip install .\
    && cd / \
    && rm -rf lcm

# Install FR3Py
RUN git clone --recursive https://github.com/Rooholla-KhorramBakht/FR3Py.git \
    && cd FR3Py \
    && pip install -e .

# Install FR3Py-Gripper
RUN cd FR3Py/fr3_gripper \
    && pip install .
    
# Install C++ Bridge
RUN cd FR3Py/fr3_bridge \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make -j $(( $(nproc) - 1 )) \
    && make \
    && make install 

# Clean up
RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf ~/.cache/pip

# Spin the container
CMD ["tail", "-f", "/dev/null"]