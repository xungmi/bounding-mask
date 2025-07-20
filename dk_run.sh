#!/bin/bash

docker run --gpus all -d \
  --runtime=nvidia \
  --net=host \
  --privileged \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/app \
  --ipc=host \
  --name gsav2 \
  gsa:v2 \
  /bin/bash
