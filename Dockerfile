FROM nvidia/cudagl:11.3.1-devel-ubuntu20.04 AS runtime
ENV DEBIAN_FRONTEND=nonintercative
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
WORKDIR /root

RUN apt-get update
RUN apt-get install -y wget nano htop tmux libsparsehash-dev rsync wget libsm6 libxext6 git rsync sudo ssh vim nano unzip zip pv gcp byobu
RUN mkdir -p /usr/share/vulkan/icd.d
RUN echo '{"file_format_version": "1.0.0", "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.1.84"}}' > /usr/share/vulkan/icd.d/nvidia_icd.json

RUN wget \
    https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py38_4.11.0-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-py38_4.11.0-Linux-x86_64.sh -b \
    && rm -f Miniconda3-py38_4.11.0-Linux-x86_64.sh

RUN conda --version
COPY environment.yml .
RUN echo \
'channels: \n\
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ \n\
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ \n\
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ \n\
ssl_verify: true' > ~/.condarc
SHELL ["/bin/bash", "-c"]
RUN  conda env create -f environment.yml  && conda clean --yes --all
# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "activezero2", "/bin/bash", "-c"]
RUN pip --no-cache-dir install path.py tqdm open3d opencv-contrib-python tabulate transforms3d https://storage1.ucsd.edu/wheels/sapien-dev/sapien-2.0.0.dev20220412-cp38-cp38-manylinux2014_x86_64.whl scikit-image