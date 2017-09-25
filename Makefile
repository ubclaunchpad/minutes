# Image names
PROD_IMAGE_NAME=minutes-prod
DEV_IMAGE_NAME=minutes-dev

# Ports
PROD_PORT=80
DEV_PORT=8080

.PHONY: dev

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
	docker build -t $(PROD_IMAGE_NAME) .

build-dev:
	docker build \
			-f ./dev/Dockerfile \
			-t $(DEV_IMAGE_NAME) .
