import gymnasium as gym 
from stable_baselines3 import PPO, A2C, SAC, DQN
from Params import Params
from EnvWrapper import ActionWrapper
from gymnasium.wrappers import RecordVideo, HumanRendering

STD_PARAMS = Params('../params/model.yaml')
ALGS = {'PPO': PPO, 'A2C': A2C, 'SAC': SAC, 'DQN': DQN}

class Agent():

    def __init__(self, params: Params = STD_PARAMS):
        self.model_params = params
        self.env = self.set_env(self.model_params)
        self.model = self.set_model(self.model_params)
        
    
    def train(self, params: Params):

        self.model.learn(
            total_timesteps=params.train['total_timesteps'],
            log_interval=params.train['log_interval'],
            tb_log_name=self.model_params.alg['algorithm'],
            progress_bar=params.train['progress_bar']
                         )
        self.model.save(f"{params.train['path_to_models']}/{self.model_params.alg['algorithm']}_{params.train['total_timesteps']}"+
                        f"{self.model_params.train['learning_rate']}")

    
    def eval(self, params: Params, render_mode: str):

        extra_steps = params.eval['extra_steps']
        episodes = params.eval['episodes']
        self.env = self.set_env(params)
        if (render_mode == 'human'):
            self.env = HumanRendering(self.env)
        else:
            self.env = RecordVideo(self.env, video_folder=params.eval['video_folder'], 
                                   name_prefix=f"{params.eval['model']}",
                                    episode_trigger=lambda x: True)
            
        self.model = ALGS[params.alg['algorithm']].load(params.eval['path_to_models']+params.eval['model'], env=self.env)

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
    
    def set_model(self, params):
        if params.alg['algorithm'] == 'PPO':
            return PPO(
                policy=params.alg['policy'],
                env=self.env,
                learning_rate=params.PPO['learning_rate'],
                n_steps=params.PPO['n_steps'],
                batch_size=params.PPO['batch_size'],
                n_epochs=params.PPO['n_epochs'],
                gamma=params.PPO['gamma'],
                gae_lambda=params.PPO['gae_lambda'],
                clip_range=params.PPO['clip_range'],
                clip_range_vf=params.PPO['clip_range_vf'],
                normalize_advantage=params.PPO['normalize_advantage'],
                ent_coef=params.PPO['ent_coef'],
                vf_coef=params.PPO['vf_coef'],
                max_grad_norm=params.PPO['max_grad_norm'],
                use_sde=params.PPO['use_sde'],
                sde_sample_freq=params.PPO['sde_sample_freq'],
                rollout_buffer_class=params.PPO['rollout_buffer_class'],
                rollout_buffer_kwargs=params.PPO['rollout_buffer_kwargs'],
                target_kl=params.PPO['target_kl'],
                stats_window_size=params.PPO['stats_window_size'],
                tensorboard_log=params.alg['tensorboard_log'],
                policy_kwargs=params.PPO['policy_kwargs'],
                verbose=params.alg['verbose'],
                seed=params.alg['seed'],
                device=params.alg['device']
            )
        elif params.alg['algorithm'] == 'DQN':
            return DQN(
                policy=params.alg['policy'],
                env=self.env,
                learning_rate=params.DQN['learning_rate'],
                buffer_size=params.DQN['buffer_size'],
                learning_starts=params.DQN['learning_starts'],
                batch_size=params.DQN['batch_size'],
                tau=params.DQN['tau'],
                gamma=params.DQN['gamma'],
                train_freq=params.DQN['train_freq'],
                gradient_steps=params.DQN['gradient_steps'],
                replay_buffer_class=params.DQN['replay_buffer_class'],
                replay_buffer_kwargs=params.DQN['replay_buffer_kwargs'],
                optimize_memory_usage=params.DQN['optimize_memory_usage'],
                target_update_interval=params.DQN['target_update_interval'],
                exploration_fraction=params.DQN['exploration_fraction'],
                exploration_initial_eps=params.DQN['exploration_initial_eps'],
                exploration_final_eps=params.DQN['exploration_final_eps'],
                max_grad_norm=params.DQN['max_grad_norm'],
                stats_window_size=params.DQN['stats_window_size'],
                tensorboard_log=params.alg['tensorboard_log'],
                policy_kwargs=params.DQN['policy_kwargs'],
                verbose=params.alg['verbose'],
                device=params.alg['device']
            )
        elif params.alg['algorithm'] == 'SAC':
            return SAC(
                policy=params.alg['policy'],
                env=self.env,
                learning_rate=params.SAC['learning_rate'],
                buffer_size=params.SAC['buffer_size'],
                learning_starts=params.SAC['learning_starts'],
                batch_size=params.SAC['batch_size'],
                tau=params.SAC['tau'],
                gamma=params.SAC['gamma'],
                train_freq=params.SAC['train_freq'],
                gradient_steps=params.SAC['gradient_steps'],
                action_noise=params.SAC['action_noise'],
                replay_buffer_class=params.SAC['replay_buffer_class'],
                replay_buffer_kwargs=params.SAC['replay_buffer_kwargs'],
                optimize_memory_usage=params.SAC['optimize_memory_usage'],
                ent_coef=params.SAC['ent_coef'],
                target_update_interval=params.SAC['target_update_interval'],
                target_entropy=params.SAC['target_entropy'],
                use_sde=params.SAC['use_sde'],
                sde_sample_freq=params.SAC['sde_sample_freq'],
                use_sde_at_warmup=params.SAC['use_sde_at_warmup'],
                stats_window_size=params.SAC['stats_window_size'],
                tensorboard_log=params.alg['tensorboard_log'],
                policy_kwargs=params.SAC['policy_kwargs'],
                verbose=params.alg['verbose'],
                seed=params.alg['seed'],
                device=params.alg['device']
            )
        elif params.alg['algorithm'] == 'A2C':
            return A2C(
                policy=params.alg['policy'],
                env=self.env,
                learning_rate=params.A2C['learning_rate'],
                n_steps=params.A2C['n_steps'],
                gamma=params.A2C['gamma'],
                gae_lambda=params.A2C['gae_lambda'],
                ent_coef=params.A2C['ent_coef'],
                vf_coef=params.A2C['vf_coef'],
                max_grad_norm=params.A2C['max_grad_norm'],
                rms_prop_eps=params.A2C['rms_prop_eps'],
                use_rms_prop=params.A2C['use_rms_prop'],
                use_sde=params.A2C['use_sde'],
                sde_sample_freq=params.A2C['sde_sample_freq'],
                rollout_buffer_class=params.A2C['rollout_buffer_class'],
                rollout_buffer_kwargs=params.A2C['rollout_buffer_kwargs'],
                normalize_advantage=params.A2C['normalize_advantage'],
                stats_window_size=params.A2C['stats_window_size'],
                tensorboard_log=params.alg['tensorboard_log'],
                policy_kwargs=params.A2C['policy_kwargs'],
                verbose=params.alg['verbose'],
                seed=params.alg['seed'],
                device=params.alg['device']
            )
        else:
            raise ValueError(f"Unsupported algorithm: {params.alg['algorithm']}")
    
    def set_env(self, params):
        if params.alg['algorithm'] == 'PPO':
            return ActionWrapper(gym.make('CarRacing-v2', continuous=True, render_mode='rgb_array'))
        elif params.alg['algorithm'] == 'DQN':
            return gym.make('CarRacing-v2', continuous=False, render_mode='rgb_array')
        elif params.alg['algorithm'] == 'SAC':
            return ActionWrapper(gym.make('CarRacing-v2', continuous=True, render_mode='rgb_array'))
        elif params.alg['algorithm'] == 'A2C':
            return ActionWrapper(gym.make('CarRacing-v2', continuous=True, render_mode='rgb_array'))
        else:
            raise ValueError(f"Unsupported algorithm: {params.alg['algorithm']}")