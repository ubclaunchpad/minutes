# Image names
PROD_IMAGE_NAME=minutes-prod
DEV_IMAGE_NAME=minutes-dev
ORGANIZATION=ubclaunchpad
TAG=latest

# Finding containers.
DEV_CONTAINER=`docker ps -aq --filter name=$(DEV_IMAGE_NAME)`
PROD_CONTAINER=`docker ps -aq --filter name=$(PROD_IMAGE_NAME)`

# Repository information.
DEV_LOCAL=$(DEV_IMAGE_NAME):$(TAG)
PROD_LOCAL=$(PROD_IMAGE_NAME):$(TAG)
DEV_REMOTE=$(ORGANIZATION)/$(DEV_LOCAL)
PROD_REMOTE=$(ORGANIZATION)/$(PROD_LOCAL)

# Ports
PROD_PORT=8081
DEV_PORT=8080

# Digital ocean (set this env variable MINUTES_RESEARCH_INSTANCE).
RESEARCH_INSTANCE=${MINUTES_RESEARCH_INSTANCE}
REMOTE_DATA_FOLDER=/research/nb/data

.PHONY: all dev run build-prod build-dev push-dev push-prod

all: build-prod build-dev

run:
	docker rm -f $(PROD_CONTAINER) 2>> /dev/null || true
	docker run -it \
		--name $(PROD_IMAGE_NAME) \
		-p $(PROD_PORT):$(PROD_PORT) \
		$(PROD_IMAGE_NAME)

dev:
	@echo "Opening dev environment on localhost port $(DEV_PORT)."
	docker rm -f $(DEV_CONTAINER) 2>> /dev/null || true
	docker run --rm -d \
		--name $(DEV_IMAGE_NAME) \
		--memory=4g \
		-v `pwd`/nb:/nb \
		-v `pwd`/collector:/collector \
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
		docker tag $(DEV_LOCAL) $(DEV_REMOTE) && \
		docker push $(DEV_REMOTE)

push-prod:
	docker login -u=$(DOCKER_USERNAME) -p=$(DOCKER_PASSWORD) && \
		docker tag $(PROD_LOCAL) $(PROD_REMOTE) && \
		docker push $(PROD_REMOTE)

pull-nb:
	scp -i ~/.ssh/id_minutes -r root@$(RESEARCH_INSTANCE):/research/nb/ .

push-data:
	scp -i ~/.ssh/id_minutes -r $(FILE) \
		root@$(RESEARCH_INSTANCE):$(REMOTE_DATA_FOLDER)/$(FILE)
