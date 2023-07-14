set -e

# variables
IEDA_WORKSPACE=$(cd "$(dirname "$0")/../..";pwd)
DOCKERFILE_DIR=${IEDA_WORKSPACE}/scripts/docker
REMOTE_ID=iedaopensource

push_tag()
{
  docker tag $1 $2
  # docker push $2
}

update_img()
{
  echo "========== update_img =========="
  export DOCKER_BUILDKIT=1
  local IMG_TAG=$1 # release:latest
  local DOCKERFILE=$2 # Dockerfile.release
  local BASE_IMAGE=$3
  local RELEASE_IMAGE=$4

  if [[ ${#BASE_IMAGE} != 0 ]] ; then
    docker build \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    --build-arg RELEASE_IMAGE="${RELEASE_IMAGE}" \
    --build-arg IEDA_REPO="${IEDA_WORKSPACE}" \
    --tag   ${IMG_TAG} \
    --file  ${DOCKERFILE_DIR}/${DOCKERFILE} \
    "${DOCKERFILE_DIR}"
  else
    docker build \
    --build-arg IEDA_REPO="${IEDA_WORKSPACE}" \
    --tag   ${IMG_TAG} \
    --file  ${DOCKERFILE_DIR}/${DOCKERFILE} \
    "${DOCKERFILE_DIR}"
  fi 

  push_tag "${IMG_TAG}" "${REMOTE_ID}/${IMG_TAG}"

  # tag version "latest" is equivalent to "debian"
  if [[ ${IMG_TAG##*:} == "latest" ]]; then
    push_tag "${IMG_TAG}" "${REMOTE_ID}/${IMG_TAG/%:*/:debian}"
  fi
}

update_img base:latest    Dockerfile.base
# update_img base:ubuntu    Dockerfile.base ubuntu:20.04
# update_img release:latest Dockerfile.release
# update_img release:ubuntu Dockerfile.release iedaopensource/base:ubuntu ubuntu:20.04

#   # =========== update iedaopensource/demo:gcd ===========
#   docker build --no-cache \
#     --tag   iedaopensource/demo:gcd \
#     --file  "${DOCKERFILE_DIR}/Dockerfile.demogcd" \
#     "${DOCKERFILE_DIR}"
#   # docker push iedaopensource/demo:gcd
