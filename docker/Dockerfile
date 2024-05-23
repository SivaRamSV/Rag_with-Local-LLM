# Use a base image that supports the required dependencies
FROM centos:centos7.3.1611

# Install necessary packages, including specific version of make if needed
# Install EPEL repository for additional packages
# Install EPEL repository for additional packages
RUN yum install -y epel-release



# Install Development Tools, which includes Make
RUN yum groupinstall -y "Development Tools"

# Install Software Collections (SCL) repository
RUN yum install -y centos-release-scl

# Install Python 3.11 from the SCL repository

# Install Git
RUN yum install -y git

# Install dependencies required for building Python
RUN yum install -y \
    wget \
    gcc \
    openssl \
    openssl-devel \
    bzip2-devel \
    libffi-devel \
    readline-devel \
    sqlite-devel \
    zlib-devel \
    tk-devel \
    libuuid-devel \
    xz-devel \
    ncurses-devel \
    gdbm-devel

# Install dependencies and devtoolset
RUN yum install -y devtoolset-11-gcc devtoolset-11-gcc-c++


# Enable devtoolset
RUN scl enable devtoolset-11 bash

# Download and install OpenSSL 1.1.1
RUN wget https://ftp.openssl.org/source/openssl-1.1.1k.tar.gz --no-check-certificate  && tar xf openssl*.gz  && cd openssl*  && ./config --prefix=/usr --openssldir=/etc/ssl zlib-dynamic  && make -j$(nproc)  && make install









# Download Python 3.11 source code
RUN wget https://www.python.org/ftp/python/3.11.0/Python-3.11.0.tgz

# Extract Python source code
RUN tar xzf Python-3.11.0.tgz

# Navigate into Python source directory
WORKDIR /Python-3.11.0

# Configure Python with SSL support
RUN ./configure --with-openssl=/usr

RUN make -j$(nproc)
RUN make altinstall

# Cleanup
WORKDIR /
RUN rm -rf Python-3.11.0.tgz Python-3.11.0

# Verify Python installation
RUN python3.11 --version

# Install pip
RUN yum install -y python3-pip



RUN pip3.11 install openai 
# Enable devtoolset
RUN scl enable devtoolset-11 '''pip3.11 install 'llama-cpp-python[server]''''''
RUN pip3.11 install pydantic 
RUN pip3.11 install instructor streamlit


# Set Python 3.11 as the default Python interpreter


# Clone the llama-cpp-python repository
RUN git clone https://github.com/ggerganov/llama.cpp.git

# Set working directory
WORKDIR /llama.cpp/model

COPY ./model/mistral-7b-instruct-v0.2.Q8_0.gguf .
# Configure and build the package

# Expose any necessary ports
EXPOSE 8080

# Command to run the server (replace with actual command if needed)
# CMD ["python", "server.py"]
#python3.11 -m llama_cpp.server --model model/mixtral-8x7b-v0.1.Q5_K_M.gguf --n_gpu -1 --port 8080
# Example command to start the server (if applicable)
# CMD ["./server"]