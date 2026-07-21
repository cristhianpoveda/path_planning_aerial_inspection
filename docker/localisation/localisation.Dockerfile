# localisation service — ROS 2 Humble
FROM ros:humble-perception-jammy

ARG USERNAME=ros
ARG USER_UID=1000
ARG USER_GID=1000

SHELL ["/bin/bash", "-c"]

# --- apt deps
COPY docker/localisation/apt-localisation.txt /tmp/apt.txt
RUN apt-get update \
    && sed 's/#.*//' /tmp/apt.txt | grep -vE '^[[:space:]]*$' \
       | xargs -r apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/* /tmp/apt.txt

# ---- Pangolin (pin a release; master breaks periodically) ----
RUN git clone --depth 1 --branch v0.8 https://github.com/stevenlovegrove/Pangolin.git /opt/Pangolin \
    && cd /opt/Pangolin \
    && cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF \
    && cmake --build build -j"$(nproc)" \
    && cmake --install build \
    && ldconfig \
    && rm -rf /opt/Pangolin/build

# ---- ORB-SLAM3 (built as a shared lib, installed system-wide) ----
RUN git clone --depth 1 https://github.com/UZ-SLAMLab/ORB_SLAM3.git /opt/ORB_SLAM3 \
    && cd /opt/ORB_SLAM3 \
    && sed -i 's/++11/++14/g' CMakeLists.txt \
    && chmod +x build.sh && ./build.sh
ENV ORB_SLAM3_ROOT_DIR=/opt/ORB_SLAM3

RUN cd /opt/ORB_SLAM3/Vocabulary && tar -xzf ORBvoc.txt.tar.gz

RUN printf '%s\n' \
      /opt/ORB_SLAM3/lib \
      /opt/ORB_SLAM3/Thirdparty/DBoW2/lib \
      /opt/ORB_SLAM3/Thirdparty/g2o/lib \
      > /etc/ld.so.conf.d/orbslam3.conf \
    && ldconfig

# --- pip deps
COPY docker/localisation/requirements-localisation.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# --- non-root 'ros' user with passwordless sudo
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
