#!/usr/bin/env bash
set -e

if [[ -d "$HOME/blender" ]]; then
    blenderproc --custom-blender-path "$HOME/blender" "${@:1}"
else
    blenderproc --blender-install-path "$HOME/blender" "${@:1}"
fi
