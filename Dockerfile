FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

WORKDIR /stage

# Install base packages.
RUN apt-get update --fix-missing && apt-get install -y \
    bzip2 \
    ca-certificates \
    curl \
    gcc \
    git \
    libc-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    wget \
    libevent-dev \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Anaconda.
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# Use python 3.5
RUN conda install python=3.5

# Copy files.
COPY bilm/ bilm/
COPY bin/ bin/
COPY tests/ tests/
COPY README.md README.md
COPY setup.py setup.py
COPY usage*.py ./
COPY run_tests_before_shell.sh run_tests_before_shell.sh

# Install package.
RUN pip install tensorflow-gpu==1.2 h5py
RUN python setup.py install

# Run tests to verify the Docker build
# and then drop into a shell.
CMD ["/bin/bash", "run_tests_before_shell.sh"]
