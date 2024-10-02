# Use an Ubuntu base image
FROM ubuntu:22.04

# Set environment variables for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies
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

# # Build the EMVB project


# Download the correct tar.gz file


# Extract the downloaded file

# Set the working directory (create if it doesn't exist)
WORKDIR /index

# Download the file into the current working directory (/index)
RUN wget http://hpc.isti.cnr.it/~rulli/emvb-ecir2024/lotte/260k_m32_LOTTE_OPQ.tar.gz

# Extract the downloaded file into the current working directory (/index)
RUN tar -xvzf 260k_m32_LOTTE_OPQ.tar.gz

# Optional: Remove the tar.gz file to save space
RUN rm 260k_m32_LOTTE_OPQ.tar.gz

WORKDIR /code


RUN . /opt/intel/oneapi/setvars.sh && mkdir build && cd build \
    && cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF .. \
    && make -j


# # Set the number of threads for MKL (as recommended)
# ENV OMP_NUM_THREADS=1

# # Add an entrypoint to easily run the EMVB binary
ENTRYPOINT ["./build/perf_embv"]

