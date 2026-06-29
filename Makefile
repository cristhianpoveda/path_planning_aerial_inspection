# First run: make images && make build
export UID := $(shell id -u)
export GID := $(shell id -g)

.PHONY: images build up up-bridge up-loc shell-bridge shell-loc down clean

# Build the docker images
images:
	docker compose build

# Colcon build packages in one-off containers.
# --no-deps stops services being started as dependencies, so a build never
# leaves an extra container holding the phone's single-client slot.
build:
	docker compose run --rm --no-deps bridge bash -lc "cd /ros2_ws && colcon build --symlink-install"
	docker compose run --rm --no-deps localization bash -lc "cd /ros2_ws && colcon build --symlink-install"

# Bring containers up IDLE (detached). Nothing launches automatically; both
# services run `bash` and wait. Launch nodes yourself from the shells below.
up:
	docker compose up -d bridge localization

up-bridge:
	docker compose up -d bridge

up-loc:
	docker compose up -d localization

# Interactive shell INSIDE the already-running container (attaches; does not
# spawn a new instance). From here you run, e.g.:
#   ros2 launch drone_bridge bridge.launch.py
shell-bridge:
	docker compose exec bridge bash

# From here you run, e.g.:
#   ros2 launch drone_localization localization.launch.py mode:=mapping \
#       map_db:=/maps/site_atlas vocab:=/config/ORBvoc.txt \
#       camera_config:=/config/camera_down.yaml
shell-loc:
	docker compose exec localization bash

# Stop and delete compose network (sweeps up orphaned run containers too).
down:
	docker compose down --remove-orphans

clean:         ## remove colcon build artifacts
	docker compose run --rm --no-deps bridge bash -lc "cd /ros2_ws && rm -rf build install log"