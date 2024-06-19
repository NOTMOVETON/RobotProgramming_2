import torch as th
import numpy as np
from stable_baselines3 import PPO, SAC, A2C, DQN
import gymnasium as gym

class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(ActionWrapper, self).__init__(env)

    def action(self, action):
        return [float(i) for i in action]


env = ActionWrapper(gym.make("CarRacing-v2"))
model = PPO('CnnPolicy', env, verbose=1, device='cuda', tensorboard_log='./runs')

model.learn(total_timesteps=5000)
model_dir = './models'
sb3_algo = './PPOcarcnn'
TIMESTEPS = 5000
model.save(f"{model_dir}/{sb3_algo}_{TIMESTEPS}")

env.close()