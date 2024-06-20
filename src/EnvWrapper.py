import gymnasium as gym

class ActionWrapper(gym.ActionWrapper):
    '''
    Due to unknown reason, in docker container environment without
    a wrapper gives error(env.step(action) should get float value but get
    something else instead). So we decided to use API that gymnasium provides 
    to solve this problem. This code overrides action function in env, so we 
    explicitly get float values.
    '''
    def __init__(self, env):
        super(ActionWrapper, self).__init__(env)

    def action(self, action):
        return [float(i) for i in action]