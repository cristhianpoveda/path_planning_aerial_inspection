# Localization container: ORB-SLAM3 (L3 mapping / L4 localization) + robot_localization.
# Feature-based SLAM is CPU-bound -> no CUDA base, no --gpus flag.
#
# Dependency chain built in this exact order — each one is a prerequisite of the next:
#   Eigen3 (apt)  ->  Pangolin  ->  ORB-SLAM3 core  ->  orb_slam3_ros2_wrapper (colcon)
#
# ORB-SLAM3 bundles its own copies of DBoW2 and g2o as Thirdparty/ subdirectories and
# builds them via its own build.sh — you do NOT need to build external g2o or FBoW here.
# That was a stella_vslam requirement; ORB-SLAM3 manages its own graph-optimisation and
# bag-of-words libs internally.
#
# Base pinned to the OS-qualified -jammy tag (Ubuntu 22.04, Humble's OS).
# For full reproducibility add a digest:
#   docker buildx imagetools inspect ros:humble-perception-jammy   # get sha256
#   FROM ros:humble-perception-jammy@sha256:<digest>
FROM ros:humble-perception-jammy

ARG USERNAME=ros
ARG USER_UID=1000
ARG USER_GID=1000

SHELL ["/bin/bash", "-c"]

# ── ROS apt deps ────────────────────────────────────────────────────────────
# cv_bridge / image_transport already in the perception image.
RUN apt-get update && apt-get install -y --no-install-recommends \
      sudo \
      ros-humble-robot-localization \
      ros-humble-message-filters \
      ros-humble-tf2-ros \
      ros-humble-tf2-geometry-msgs \
    && rm -rf /var/lib/apt/lists/*

# ── ORB-SLAM3 + Pangolin system build deps (explicit; correct jammy names) ──
# Eigen 3.4.0 from apt on jammy — confirmed compatible.
# OpenCV 4.x is already in the perception image.
# Pangolin deps are installed here explicitly rather than via its
# install_prerequisites.sh script, which assumes a fresh apt cache and uses
# pre-jammy package names (e.g. libegl1-mesa-dev -> now libegl-dev).
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git pkg-config ninja-build \
      libeigen3-dev \
      libssl-dev \
      libglew-dev \
      libgflags-dev \
      libgoogle-glog-dev \
      ffmpeg \
      python3-pip \
      libgl1-mesa-dev \
      libegl-dev \
      libwayland-dev \
      libxkbcommon-dev \
      wayland-protocols \
      libavcodec-dev \
      libavutil-dev \
      libavformat-dev \
      libswscale-dev \
      libavdevice-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Pangolin (headless; viewer disabled at runtime via Viewer.UseViewer: 0) ──
# Deps installed in the apt block above; install_prerequisites.sh skipped.
COPY docker/orb_slam3.repos /tmp/orb_slam3.repos
RUN mkdir -p /opt/src && vcs import --recursive /opt/src < /tmp/orb_slam3.repos \
    && echo "=== monorepo contents ===" && ls -la /opt/src/orb_slam3_ros2_docker

RUN cd /opt/src/Pangolin \
    && git checkout v0.9.1 \
    && cmake -B build -GNinja -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j"$(nproc)" \
    && cmake --install build \
    && ldconfig

# ── ORB-SLAM3 core library ───────────────────────────────────────────────────
# The wrapper's FindORB_SLAM3.cmake HARDCODES ORB_SLAM3_ROOT_DIR=/home/orb/ORB_SLAM3
# (it ignores the env var despite the comment). Build there so CMake finds
# include/, lib/libORB_SLAM3.so, and Thirdparty/{DBoW2,g2o}/lib. This dir must
# persist in the final image — the wrapper links against it at runtime.
RUN mkdir -p /home/orb \
    && cp -r /opt/src/orb_slam3_ros2_docker/ORB_SLAM3 /home/orb/ORB_SLAM3 \
    && cd /home/orb/ORB_SLAM3 \
    && chmod +x build.sh \
    && ./build.sh \
    && echo "/home/orb/ORB_SLAM3/lib" > /etc/ld.so.conf.d/orbslam3.conf \
    && echo "/home/orb/ORB_SLAM3/Thirdparty/DBoW2/lib" >> /etc/ld.so.conf.d/orbslam3.conf \
    && echo "/home/orb/ORB_SLAM3/Thirdparty/g2o/lib" >> /etc/ld.so.conf.d/orbslam3.conf \
    && ldconfig

# ── orb_slam3_ros2_wrapper + slam_msgs (colcon overlay at /opt/orb_slam3_ros2) ──
# Both packages live as subdirectories of the suchetanrs monorepo.
# The wrapper publishes TF, handles tracking loss, and supports Atlas save/load
# for the L3 (mapping) -> L4 (localization) workflow.
RUN mkdir -p /opt/orb_slam3_ros2/src \
    && cp -r /opt/src/orb_slam3_ros2_docker/orb_slam3_ros2_wrapper \
             /opt/orb_slam3_ros2/src/ \
    && cp -r /opt/src/orb_slam3_ros2_docker/slam_msgs \
             /opt/orb_slam3_ros2/src/ \
    && source /opt/ros/humble/setup.bash \
    && cd /opt/orb_slam3_ros2 \
    && colcon build --symlink-install

# Vocabulary: ORBvoc.txt is large (~75 MB). Place it on the host under ./config
# and it will be available inside the container at /config/ORBvoc.txt.

# ── non-root user with passwordless sudo (validated drop-in) ─────────────────
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
