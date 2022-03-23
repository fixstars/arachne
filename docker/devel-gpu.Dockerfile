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

# Add a user that UID:GID will be updated by vscode
ARG USERNAME=developer
ARG GROUPNAME=developer
ARG UID=1000
ARG GID=1000
ARG PASSWORD=developer
RUN groupadd -g $GID $GROUPNAME && \
    useradd -m -s /bin/bash -u $UID -g $GID -G sudo $USERNAME && \
    echo $USERNAME:$PASSWORD | chpasswd && \
    echo "$USERNAME   ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER $USERNAME
ENV HOME /home/developer
ENV PATH $HOME/.local/bin:/usr/local/poetry/bin:$PATH

# Stage 2: including src image
# Clone src
FROM base AS src

# You need a personal access token of GitLab for cloning from gitlab,
ARG GITLAB_USERNAME=oauth2
ARG GITLAB_ACCESS_TOKEN
RUN if [[ -z "$GITLAB_ACCESS_TOKEN" ]] ; then \
    printf "\nERROR: This Dockerfile needs the personal access token of gitlab.fixstars.com, please specify by:\ndocker build --build-arg GITLAB_ACCESS_TOKEN=<your_personal_access_token>\n" && \
    exit 1; fi

RUN git clone https://${GITLAB_USERNAME}:${GITLAB_ACCESS_TOKEN}@gitlab.fixstars.com/arachne/arachne.git $HOME/arachne_src
RUN git clone --recursive https://github.com/fixstars/tvm $HOME/arachne_src/3rdparty/tvm
