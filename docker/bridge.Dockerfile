# Bridge container: DJI -> ROS layer.
# WildBridge dji_controller (HTTP telemetry) + your adapter nodes + TCP H.264 decode.
#
# Base pinned to the OS-qualified tag (-jammy = Ubuntu 22.04, Humble's OS) so the base
# OS can't shift under you. For full reproducibility, also pin the digest:
#   docker buildx imagetools inspect ros:humble-ros-base-jammy   # grab the live sha256
#   FROM ros:humble-ros-base-jammy@sha256:<digest>
FROM ros:humble-ros-base-jammy

ARG USERNAME=ros
ARG USER_UID=1000
ARG USER_GID=1000

SHELL ["/bin/bash", "-c"]

# --- system / ROS apt deps (versions are frozen by the pinned base image) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
      sudo \
      python3-pip \
      python3-numpy \
      ffmpeg \
      ros-humble-cv-bridge \
      ros-humble-image-transport \
      ros-humble-image-transport-plugins \
      ros-humble-tf2-ros \
      ros-humble-tf2-geometry-msgs \
      ros-humble-rviz2 \
    && rm -rf /var/lib/apt/lists/*

# --- pinned Python deps from a versions file ---
COPY docker/requirements-bridge.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# --- non-root user with passwordless sudo via a sudoers.d drop-in (validated; safer than
#     editing /etc/sudoers). No device groups: the drone is on the network at this stage. ---
RUN groupadd --gid ${USER_GID} ${USERNAME} 2>/dev/null || true \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash ${USERNAME} 2>/dev/null || true \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME} \
    && visudo -cf /etc/sudoers.d/${USERNAME}

COPY docker/entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod 0755 /entrypoint.sh

USER ${USERNAME}
WORKDIR /ros2_ws
ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
CMD ["bash"]
