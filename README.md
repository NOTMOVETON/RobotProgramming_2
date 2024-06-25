# RL Racing Car project 

## Team structure
- Kotov Dmitriy, 1st year ITMO master in Robotics and AI, R4135c. [Github](https://github.com/NOTMOVETON), [Telegram](https://t.me/moveton40)
- Artem Zubko, 1st year ITMO master in Robotics and AI, R4135c. [Github](https://github.com/Artemkazub), [Telegram](https://t.me/zubko_artem)

## Project Description
Our task for whis project is to provide RL-based solution to given [envinronment (RacingCar-v2)](https://gymnasium.farama.org/environments/box2d/car_racing/). 

We have a racing car that drive along random generated track, our task is to maximaze reward on random genarated path and finish the track. 

![123](https://gymnasium.farama.org/_images/car_racing.gif)

More information about env given by the link above.

## Project Structure

- `/models`: Folder containing trained models with differnet hyper-parameters.

- `/runs`: Folder created to visualize training process with Tensorboard

- `/params`: Folder holding hyper-parameters of the models.

- `/src`: Main folder with python code for training and visualizing.

- `/videos`: Result of model training, saved as video of agent's work process.

## Project Stack
1. [Numpy](https://numpy.org/)
2. [Torch](https://pytorch.org/docs/stable/torch.html)
3. [Gymnasium](https://gymnasium.farama.org/)
4. [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/index.html)

## Installation

To install and use this project, clone the repository into desired folder:

```shell
cd /<your-folder>
git clone https://github.com/NOTMOVETON/RobotProgramming_2.git
```

Аfter installation you can use the project via [a docker container](#Docker-Usage) or by creating [python virtual environment](#Python-Virtal-Environment).

## **RECOMMENDED**: Docker Usage (Ubuntu)
### Preliminary requirements
1. Install docker-engine: [Docker Engine](https://docs.docker.com/engine/install/ubuntu/).
2. Install docker-compose-plugin: [Docker Compose](https://docs.docker.com/compose/install/linux/).
3. If you want to use graphics card (*NVIDIA ONLY*) usage install nvidia-container-toolkit: [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
4. Add docker user to your group:
```shell
sudo groupadd docker 
sudo usermod -aG docker $USER 
newgrp docker
```

### Launch container 

1. Create a few terminals (VSCode is welcome to work with docker).
2. Go to project directory:
```shell
cd /<your-folder>/RobotProgramming_2
```
3. Build docker image:
```shell
docker build -t racing_car_rl .
```
4. Visuals.
   After complete building give the rights to connect the root user to the host display:
  ```shell
  xhost +local:docker
  ```
5. Launch container.
   **RECOMMENDED**:
   if you want to use nvidia graphics card:
  ```shell
  docker compose -f docker-compose.nvidia.yml up
  ```
  or using only CPU:
  ```shell
  docker compose -f docker-compose.yml up
  ```
  After that you should see similar result:
  ```shell
  [+] Running 1/1
  ✔ Container py_rl  
  Recreated     0.9s 
  Attaching to py_rl
  ```
6. Get into container via terminal in a new terminal session(in same working directory). (opens container's new bash session)
  ```shell
  docker exec -it py_rl /bin/bash
  ```
After this step docker container is ready to work and you can go to [usage section](#Usage).

#### Docker structure

In docker container project lays in `/RacingCarRL` directory, so all of actions should be executed in this folder.

## Python Virtal Environment

Other way to use the project is by creating virtual environment with requiered modules.
Detailed guide on the deployment and functions of the virtual environment can be found at the [link](https://docs.python.org/3/library/venv.html)

Follow below steps to create venv and install necessary python packages:
1. Create venv:
```shell
python -m venv /<path-to-your-venv>
```
2. Log in to the virtual environment:
```shell
source /<path-to-your-venv>/bin/activate
```
3. Install necessary modules:
```shell
pip install -r /<your-folder>/RobotProgramming_2/requirements.txt
```

Venv now ready to work and you can go to [usage section](#Usage).

## Usage
1. Setup params files in `/params` directory.
- `model.yaml` contain general hyperparameters for algorithm (learning rate etc.) (list of parameters not full, because different algorithms takes different arguments and hyperparameters)
- `train.yaml` contain parameters for `model.learn()` method and folder for saving trained models. More information could be found [here](https://stable-baselines3.readthedocs.io/en/master/modules/base.html)
- `eval.yaml` contain parameters for evaluating model, such as folders for videos etc.   

2. Start training/evaluating models with command below:
```shell
python car_goes_brr.py -a=<action>
```
, where argument `action` can take following values: `train`, `evaluate_human` and `evaluate_record`.
While executing given action programm takes parameters from params file. 

3. (OPTIONAL) Launch `Tensorboard` by going to `http://localhost:6006` or simply use VSCode extension(RECOMMENDED).
During the training process, you can observe graphs of the average length of an episode and the average reward per episode via [tensorboard](https://pytorch.org/docs/stable/tensorboard.html).

4. (OPTIONAL) Use different algorithms. 
Via `model.yaml` you can choose algorithm for solving current task.

## Results
To verify different approaches we conducted experiments with 3 different algorithms: PPO, SAC, A2C, and different hyperparameters for each.
All of used algorithms are very different and detailed analysis on each one of them would consume too much time. For this reason, theoretical information and analysis of algorithms can be found at the [link](https://stable-baselines3.readthedocs.io/en/master/modules/base.html). (if necessary, they can be described in more detail at the face-to-face defense).

In general, PPO showed himself the best of all among A2C, PPO, DQN. Average reward after 1000 episodes of training(1000 timesteps each, 1_000_000 total) for PPO is 751, A2C- 434, DQN - 117. The reward is -0.1 every frame and +1000/N for every track tile visited, where N is the total number of tiles visited in the track. For example, if you have finished in 732 frames, your reward is 1000 - 0.1*732 = 926.8 points, so the maximum reward is tend to 1000. 

More detailed results will be shown in face2face defense of the project.
