#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-serdadev/jamba2-cpt}"
IMAGE_TAG="${IMAGE_TAG:-cu121}"
FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

echo "==> Building ${FULL_IMAGE}"
docker build -t "${FULL_IMAGE}" .

echo "==> Pushing ${FULL_IMAGE}"
docker push "${FULL_IMAGE}"

echo "==> Done"
echo "Image: ${FULL_IMAGE}"
