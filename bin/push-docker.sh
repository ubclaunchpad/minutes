#!/bin/bash
ORG=ubclaunchpad
TAG=latest
IMAGE=$1

LOCAL=$IMAGE:$TAG
REMOTE=$ORG/$IMAGE:$TAG

docker tag $LOCAL $REMOTE
docker push $REMOTE