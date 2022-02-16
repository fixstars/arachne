# Dockerfiles for arachne

## `base-gpu-jp4x.Dockerfile`

A dockerfile for creating an JetPack compatible environment including cuda, cudnn, and tensorrt.

###  For JetPack 4.6 (default)

+ cuda: 10.2
+ cudnn: 8.2.1
+ tensorrt: 8.0.1

```sh
$ pwd
/path/to/arachne

$ docker build -t {tag} -f docker/base-gpu-jp4x.Dockerfile docker
```

### For JetPack 4.5

+ cuda: 10.2
+ cudnn: 8.0.0
+ tensorrt: 7.1.3

```sh
$ pwd
/path/to/arachne

$ docker build -t {tag} -f docker/base-gpu-jp4x.Dockerfile --build-arg CUDNN=8.0.0.180-1 --build-arg LIBNVINFER=7.1.3-1 LIBNVINFER_MAJOR_VERSION=7 docker
```

## `devel-gpu.Dockerfile`

A dockerfile that provide an development environment for arachne.
This is based on the image created by `base-gpu-jp4x.Dockerfile`.
In addition, some libraries and packages will be installed for development.