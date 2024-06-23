LOCAL_BASE_IMAGE:=ubuntu:20.04
DOCKERFILE_DIRECTORY := ./docker
DOCKERCOMPOSE_FILE := docker-compose.yml
DOCKER_IMAGE_NAME:=tello-hoop
CONTAINER_NAME:=tello-hoop
PROJECT:=det 


.PHONY: build_docker
build_docker:
	docker-compose -p ${PROJECT} -f $(DOCKERFILE_DIRECTORY)/$(DOCKERCOMPOSE_FILE) build

.PHONY: run_docker
run_docker:
	xhost + &&\
	docker-compose -p ${PROJECT} -f $(DOCKERFILE_DIRECTORY)/$(DOCKERCOMPOSE_FILE) up -d &&\
	docker exec -it ${CONTAINER_NAME} bash