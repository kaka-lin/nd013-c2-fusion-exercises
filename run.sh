#!/bin/bash

docker run \
    -it \
    --gpus all \
    -e DISPLAY=$DISPLAY \
    --network=host \
    --shm-size="20g" \
    --volume="$PWD:/app/project/" \
    project2-dev bash
