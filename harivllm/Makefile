# Docker commands for nano-vllm project

# Variables
IMAGE_NAME := nano-vllm
CONTAINER_NAME := nano-vllm-container
COMPOSE_FILE := docker/docker-compose.yml
CUDA_VISIBLE_DEVICES := 0

# Build Docker image
build:
	docker build -t $(IMAGE_NAME) .

# Build with no cache
build-no-cache:
	docker build --no-cache -t $(IMAGE_NAME) .

# Run container with volume mounts
run:
	docker run -it --rm --gpus all \
		--name $(CONTAINER_NAME) \
		-v $(shell pwd):/workspace \
		-v /scratch:/data/modelpreload \
		--workdir /workspace \
		-p 8990:8990 \
		-p 8888:8888 \
		-p 6006:6006 \
		-p 5678:5678 \
		-e CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
		$(IMAGE_NAME)

# Run container in background
run-detached:
	docker run -d --gpus all \
		--name $(CONTAINER_NAME) \
		-v $(shell pwd):/workspace \
		-v /scratch:/data/modelpreload \
		--workdir /workspace \
		-p 8990:8990 \
		-p 8888:8888 \
		-p 6006:6006 \
		-p 5678:5678 \
		-e CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
		$(IMAGE_NAME)

attach:
	docker attach --detach-keys="ctrl-q" -it $(CONTAINER_NAME)

# Docker Compose commands
compose-up:
	cd docker && docker-compose up -d

compose-down:
	cd docker && docker-compose down

compose-build:
	cd docker && docker-compose build

compose-rebuild:
	cd docker && docker-compose build --no-cache

compose-logs:
	cd docker && docker-compose logs -f

# Exec into running container
exec:
	docker exec -it $(CONTAINER_NAME) /bin/bash

# Exec into compose container
exec-compose:
	cd docker && docker-compose exec llm-tutorial /bin/bash

# Alternative exec with different shell
exec-zsh:
	docker exec -it $(CONTAINER_NAME) /bin/zsh

# Stop and remove container
stop:
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true

# Clean up images and containers
clean:
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true
	docker rmi $(IMAGE_NAME) || true

# Show running containers
ps:
	docker ps

# Show container logs
logs:
	docker logs -f $(CONTAINER_NAME)

# Help
help:
	@echo "Available commands:"
	@echo "  build              - Build Docker image"
	@echo "  build-no-cache     - Build Docker image without cache"
	@echo "  run                - Run container interactively with volume mounts"
	@echo "  run-detached       - Run container in background"
	@echo "  compose-up         - Start services with docker-compose"
	@echo "  compose-down       - Stop services with docker-compose"
	@echo "  compose-build      - Build services with docker-compose"
	@echo "  compose-rebuild    - Rebuild services without cache"
	@echo "  compose-logs       - Show docker-compose logs"
	@echo "  exec               - Exec into running container"
	@echo "  exec-compose       - Exec into docker-compose container"
	@echo "  stop               - Stop and remove container"
	@echo "  clean              - Clean up containers and images"
	@echo "  ps                 - Show running containers"
	@echo "  logs               - Show container logs"
	@echo "  help               - Show this help message"

.PHONY: build build-no-cache run run-detached compose-up compose-down compose-build compose-rebuild compose-logs exec exec-compose exec-zsh stop clean ps logs help
