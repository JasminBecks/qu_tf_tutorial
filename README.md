# Quantum Universe - TensorFlow Tutorial [![Deploy images](https://github.com/riga/qu_tf_tutorial/workflows/Deploy%20images/badge.svg)](https://github.com/riga/qu_tf_tutorial/actions?query=workflow%3A%22Deploy+images%22)

### Starting the notebook

There are two methods to start the exercise notebook, both of which require [docker](https://www.docker.com/get-started) to be installed on your system.


#### 1. Docker image with a local checkout (recommended for working on your local machine)

To start the exercise in a dedicated docker image that contains all the software you need, you should clone this repostory and start the container with the helper script located at [docker/run.sh](docker/run.sh).

```shell
git clone https://github.com/riga/qu_tf_tutorial
cd qu_tf_tutorial
./docker/run.sh
```

The script will start the container with your local repostiroy mounted into it so that changes you make to the notebook persist when the container stops.

Make sure **not** to execute any command with `sudo` as a port will be opened on your machine to run and host the notebook server.
Otherwise, you potentially allow people within your local network to access your system with root permissions!

Then, open a web browser at the displayed location and click on `exercise.ipynb` to access the notebook.

**Note on docker on Linux**:
If you installed docker for the first time to run this exercise, and you appear to miss the permission to execute docker with your user account, add yourself to the "docker" group (e.g. via `sudo usermod -a -G docker $(whoami)`).

**Note on docker on Mac**:
By default, docker for Mac limits the memory available to running containers to 2 GB.
Depending on how much input data you use in the exercise (this can be configured in the notebook), you might want to increase this number to about 6 to 8 GB, depending on your system.
Instructions to change this setting can be found [here](https://docs.docker.com/docker-for-mac/#resources).


#### 2. Standalone docker image from the docker hub

If you just want to give the exercise a go without cloning a repository, you can run the docker image in a standalone fashion.
Just keep in mind that any changes you make to the notebook are stored **only within the container** and, without any further action, might not persist after the container process exits.

As above, make sure not to run the container as root!

```shell
docker run -ti -u $(id -u):$(id -g) -p 8888:8888 riga/qu_tf_tutorial
```
