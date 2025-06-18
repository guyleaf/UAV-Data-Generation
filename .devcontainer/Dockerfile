FROM mcr.microsoft.com/devcontainers/python:1-3.10-bullseye
HEALTHCHECK NONE

ARG PERSISTENCE_DIR="/persistence"
ARG BASH_HISTORY="${PERSISTENCE_DIR}/.bash_history"
ARG XDG_CACHE_HOME="${PERSISTENCE_DIR}/.cache"
ARG DEBIAN_FRONTEND="noninteractive"
ARG TZ="Asia/Taipei"

WORKDIR /tmp

# SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Remove default vscode user and replace apt mirror site
# RUN deluser vscode --remove-all-files \
#     && sed -i -e "s/http:\/\/archive\.ubuntu\.com\/ubuntu\//mirror:\/\/mirrors\.ubuntu\.com\/mirrors\.txt" /etc/apt/sources.list

RUN apt-get update && apt-get -y upgrade \
    && apt-get -y install --no-install-recommends man manpages \
    && apt-get -y install --no-install-recommends openmpi-bin openmpi-common openmpi-doc \
    && apt-get -y install libxxf86vm1 libx11-6 libxi6 libxrandr2 libxcursor1 libgl1 libxinerama1 libxkbcommon0 libsm6 \
    && apt-get -y install --no-install-recommends tmux \
    && apt-get -y install --no-install-recommends --reinstall bash-completion \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Install nvitop for monitoring GPU
RUN python3 -m pip install --no-cache-dir -U nvitop

# Persist bash history and others
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=${BASH_HISTORY}" \
    && echo "$SNIPPET" >> "$HOME/.bashrc" \
    && echo "export XDG_CACHE_HOME=${XDG_CACHE_HOME}" >> "$HOME/.bashrc"
