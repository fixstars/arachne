#!/bin/sh -eu

if [ -z "${CI:-}" ]; then
    echo "Please run this script from GitLab CI"
    exit 1
fi

# Check changes
if ! git diff --name-only $CI_COMMIT_BEFORE_SHA | grep -xFq \
    -e "3rdparty/tvm" \
    -e ".gitlab-ci.yml" \
    -e "docker/Dockerfile.tvm_rpc_tracker" \
    -e "ci/build_rpc_tracker.sh" \
    ; then
    echo "Skipped (no changes)"
    exit 0
fi

if [ "$CI_COMMIT_BRANCH" = "$CI_DEFAULT_BRANCH" ]; then
    IMAGE_TAG=latest
else
    IMAGE_TAG=$CI_COMMIT_REF_SLUG
fi
IMAGE_NAME=$CI_REGISTRY_IMAGE/tvm_rpc_tracker:$IMAGE_TAG

docker build -t $IMAGE_NAME -f docker/Dockerfile.tvm_rpc_tracker .
docker push $IMAGE_NAME
