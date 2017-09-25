# Image names
PROD_IMAGE_NAME=minutes-prod
DEV_IMAGE_NAME=minutes-dev
ORGANIZATION=ubclaunchpad
TAG=latest

# Ports
PROD_PORT=80
DEV_PORT=8080

.PHONY: dev run build-prod build-dev push-dev push-prod

all: build-prod build-dev

run:
	docker run -it \
			-p $(PROD_PORT):$(PROD_PORT) \
			$(PROD_IMAGE_NAME)

dev:
	echo "Opening dev environment on localhost port $(DEV_PORT)."
	docker run -d \
			-v `pwd`/nb:/nb \
			-p $(DEV_PORT):$(DEV_PORT) \
			$(DEV_IMAGE_NAME)

build-prod:
	docker build --rm -t $(PROD_IMAGE_NAME) .

build-dev:
	docker build --rm \
			-f ./dev/Dockerfile \
			-t $(DEV_IMAGE_NAME) .

push-dev:
		docker login -u=$(DOCKER_USERNAME) -p=$(DOCKER_PASSWORD) && \
    	docker tag $(DEV_IMAGE_NAME):$(TAG) $(ORGANIZATION)/$(DEV_IMAGE_NAME):$(TAG) && \
    	docker push $(ORGANIZATION)/$(DEV_IMAGE_NAME):$(TAG)

push-prod:
		docker login -u=$(DOCKER_USERNAME) -p=$(DOCKER_PASSWORD) && \
    	docker tag $(DEV_IMAGE_NAME):$(TAG) $(ORGANIZATION)/$(DEV_IMAGE_NAME):$(TAG) && \
    	docker push $(ORGANIZATION)/$(DEV_IMAGE_NAME):$(TAG)