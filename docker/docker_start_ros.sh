
#!/usr/bin/env bash

BASH_OPTION=bash

IMG=richardrl/bandu_v1:latest
containerid=$(docker ps -qf "ancestor=${IMG}") && echo $containerid

xhost +

if [[ -n "$containerid" ]]
then
    docker exec -it \
        --privileged \
        -e DISPLAY=${DISPLAY} \
        -e LINES="$(tput lines)" \
        stable_plane_ros \
        $BASH_OPTION
else
    docker start -i stable_plane_ros
fi
