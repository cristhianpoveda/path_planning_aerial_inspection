# navigation service — ROS 2 Humble
FROM ros:humble-ros-base-jammy@sha256:afb40d6be65331c20a114d4e229a7ef099fed1b17bf6370daee193514b32aa16

ARG USERNAME=ros
ARG USER_UID=1000
ARG USER_GID=1000

SHELL ["/bin/bash", "-c"]

# --- apt deps
COPY docker/navigation/apt-navigation.txt /tmp/apt.txt
RUN apt-get update \
    && sed 's/#.*//' /tmp/apt.txt | grep -vE '^[[:space:]]*$' \
       | xargs -r apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/* /tmp/apt.txt

# --- pip deps
COPY docker/navigation/requirements-navigation.txt /tmp/requirements.txt
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
