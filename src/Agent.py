import gymnasium as gym 
from stable_baselines3 import PPO, A2C, SAC
from Params import Params
from EnvWrapper import ActionWrapper
from gymnasium.wrappers import RecordVideo

STD_PARAMS = Params('../params/model.yaml')
ALGS = {'PPO': PPO, 'A2C': A2C, 'SAC': SAC}

class Agent():

    def __init__(self, params: Params = STD_PARAMS):
        self.model_params = params
        self.env = ActionWrapper(gym.make('CarRacing-v2'))
        self.algorithm = ALGS[params.alg['algorithm']]
        self.model = self.algorithm(
            'CnnPolicy', 
            self.env, 
            learning_rate=params.alg['learning_rate'],
            #n_steps=params.alg['n_steps'],
            #batch_size=params.alg['batch_size'],
            #n_epochs=params.alg['n_epochs'],
            #gamma=params.alg['gamma'], 
            #gae_lambda=params.alg['gae_lambda'],
            #clip_range=params.alg['clip_range'], 
            #clip_range_vf=params.alg['clip_range_vf'], 
            #normalize_advantage=params.alg['normalize_advantage'], 
            #ent_coef=params.alg['ent_coef'], 
            #vf_coef=params.alg['vf_coef'], 
            #max_grad_norm=params.alg['max_grad_norm'], 
            #use_sde=params.alg['use_sde'], 
            #sde_sample_freq=params.alg['sde_sample_freq'], 
            #rollout_buffer_class=params.alg['rollout_buffer_class'], 
            #rollout_buffer_kwargs=params.alg['rollout_buffer_kwargs'], 
            #target_kl=params.alg['target_kl'], 
            #stats_window_size=params.alg['stats_window_size'], 
            tensorboard_log=params.alg['tensorboard_log'], 
            #policy_kwargs=params.alg['policy_kwargs'], 
            verbose=params.alg['verbose'], 
            seed=params.alg['seed'], 
            device=params.alg['device'],
            #_init_setup_model=params.alg['_init_setup_model']
            )
        #print(self.model.get_parameters())
    
    def train(self, params: Params):

        self.model.learn(
            total_timesteps=params.train['total_timesteps'],
            log_interval=params.train['log_interval'],
            tb_log_name=self.model_params.alg['algorithm'],
            progress_bar=params.train['progress_bar']
                         )
        
        self.model.save(f"{params.train['path_to_models']}/{self.model_params.alg['algorithm']}_{params.train['total_timesteps']}_{self.model_params.alg['learning_rate']}")

    
    def eval(self, params: Params, render_mode: str):

        extra_steps = params.eval['extra_steps']
        episodes = params.eval['episodes']

        if (render_mode == 'human'):
            self.env = ActionWrapper(gym.make('CarRacing-v2', render_mode='human'))
        else:
            self.env = ActionWrapper(gym.make('CarRacing-v2', render_mode='rgb_array'))
            self.env = RecordVideo(self.env, video_folder=params.eval['video_folder'], 
                                   name_prefix=f"{params.eval['model']}",
                                    episode_trigger=lambda x: True)
            
        self.model = self.algorithm.load(params.eval['path_to_models']+params.eval['model'], env=self.env)

        for _ in range(episodes):
            obs, _ = self.env.reset()
            fl = False
            while True:
                action, _ = self.model.predict(obs)
                obs, _, terminated, truncated, _ = self.env.step(action)
                if (terminated or truncated) or (fl):
                    extra_steps -= 1
                    if (not fl):
                       fl = True
                    if extra_steps <= 0:
                        self.env.close()
                        break
        self.env.close()