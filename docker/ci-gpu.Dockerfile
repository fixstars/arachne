FROM arachnednn/arachne:base-gpu-jp46 as base

ENV LANG C.UTF-8
ENV PYTHONIOENCODING=utf-8

# Install other packages for development

RUN echo deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-11 main >> /etc/apt/sources.list.d/llvm.list \
    && echo deb-src http://apt.llvm.org/bionic/ llvm-toolchain-bionic-11 main >> /etc/apt/sources.list.d/llvm.list \
    && apt-key adv --fetch-keys http://apt.llvm.org/llvm-snapshot.gpg.key \
    && apt-get update && apt-get install -y llvm-11 clang-11

RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    libopenblas-dev \
    libgl1-mesa-dev \
    sudo \
    curl \
    git

# python -> python3
RUN ln -s $(which python3) /usr/local/bin/python
RUN ln -s $(which pip3) /usr/local/bin/pip

# install poetry
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | POETRY_HOME=/usr/local/poetry python - --version 1.2.0a2
ENV PATH /usr/local/poetry/bin:$PATH

COPY . /workspaces/arachne
WORKDIR /workspaces/arachne

ENV PYTHONPATH=/workspaces/arachne/python
ENV TVM_LIBRARY_PATH=/workspaces/arachne/build/tvm

RUN git clone --recursive https://github.com/fixstars/tvm 3rdparty/tvm
RUN poetry install
RUN poetry run ./scripts/install_tvm.sh
RUN poetry run ./scripts/install_torch2trt.sh

ENTRYPOINT [ "poetry", "run", "pytest", "tests", "--forked" ]