#!/usr/bin/env bash

ENV_DIR="env"
YEAR=2023
USERS=(
    "ikaruswill.1"
    "ikaruswill.2"
)
CMD_ARGS=(
    --url <DOMAIN>
    --logo <LOGOPATH>
    --token <TOKEN>
    --logo-scale 0.052
    --concurrency 4
    --logo-y 2
)

if [[ ! -d env ]]; then
    echo "Python virtual environment directory not found at: ${ENV_DIR}, creating..."
    python -m venv env
    echo "Installing gitlab-skyline..."
    pip install -e .
fi

echo "Activating python virtual environment..."
source env/bin/activate
echo

set -ex

for USER in ${USERS[@]};
do
    echo "Generating GitLab Skyline for: ${USER}"
    gitlab-skyline ${CMD_ARGS[@]} ${USER} ${YEAR}
done

set +ex
