#!/usr/bin/env bash

action() {
    # determine the directy of this file (/docker) and the repo dir
    local this_file="$( [ ! -z "$ZSH_VERSION" ] && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "$this_file" )" && /bin/pwd )"
    local repo_dir="$( cd "$( dirname "$this_dir")" && /bin/pwd )"

    # create the local data directory
    local data_dir="$repo_dir/data"
    mkdir -p "$data_dir"

    # some configs
    [ -z "$DOCKER_ROOT" ] && local DOCKER_ROOT="0"
    [ -z "$DOCKER_PORT" ] && local DOCKER_PORT="8888"

    # user option for docker run, depends on whether to run as root or not
    local user_opt="-u $(id -u):$(id -g)"
    [ "$DOCKER_ROOT" = "1" ] && user_opt=""

    # run the container
    docker run \
        --rm \
        -ti \
        -w /tutorial \
        -v "$repo_dir":/tutorial \
        -v "$data_dir":/data \
        -e "NB_PORT=$DOCKER_PORT" \
        -p $DOCKER_PORT:$DOCKER_PORT \
        $user_opt \
        riga/qu_tf_tutorial $@
}
action "$@"
