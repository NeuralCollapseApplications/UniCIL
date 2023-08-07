#!/bin/bash

DATALOC=${DATALOC:-$(realpath ../datasets)}
LOGLOC=${LOGLOC:-$(realpath ../logger)}
IMG=${IMG:-"harbory/openmmlab:2206"}
GPUS=${GPUS:-"all"}

if [ "${GPUS}" != "all" ]; then
  GPUS=device="${GPUS}"
fi

docker run --gpus "${GPUS}" -it --rm --ipc=host --net=host \
  --mount src="$(pwd)",target=/opt/project,type=bind \
  --mount src="$DATALOC",target=/opt/project/data,type=bind \
  --mount src="$LOGLOC""/MMCIL/",target=/opt/logger,type=bind \
  "$IMG"
