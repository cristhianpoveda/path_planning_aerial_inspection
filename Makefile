# drone-autonomy — Docker environment
export UID := $(shell id -u)
export GID := $(shell id -g)

.PHONY: build up down ps \
        shell-bridge shell-localisation shell-navigation shell-evaluation

# Build all four service images.
build:
	docker compose build

# Bring every service up, detached and IDLE
up:
	docker compose up -d

# Stop and remove containers + network.
down:
	docker compose down --remove-orphans

ps:
	docker compose ps

# Attach an interactive shell to a running service container.
shell-bridge:
	docker compose exec bridge bash
shell-localisation:
	docker compose exec localisation bash
shell-navigation:
	docker compose exec navigation bash
shell-evaluation:
	docker compose exec evaluation bash
