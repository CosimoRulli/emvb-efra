
FROM ubuntu:22.04


ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    python3 \
    python3-pip \
    python3-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Intel MKL (Math Kernel Library)
RUN apt update | apt install -y gpg-agent wget
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor |   tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
RUN echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" |   tee /etc/apt/sources.list.d/oneAPI.listecho "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list
RUN apt update
RUN apt install -y intel-oneapi-mkl intel-oneapi-mkl-devel

RUN apt-get update && apt-get -y install cmake protobuf-compiler

RUN . /opt/intel/oneapi/setvars.sh

COPY ./ /code


WORKDIR /index


RUN wget http://hpc.isti.cnr.it/~rulli/emvb-ecir2024/efra/index_for_efra.tar.gz


RUN tar -xvzf index_for_efra.tar.gz


RUN rm index_for_efra.tar.gz

WORKDIR /code

RUN python3 -m pip install requests

RUN . /opt/intel/oneapi/setvars.sh && mkdir build && cd build \
    && cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF .. -DFAISS_OPT_LEVEL=generic \
    && make -j



RUN ls -l /code/build
RUN chmod +x /code/build/perf_emvb

WORKDIR /code


CMD ["python3", "main.py"]
