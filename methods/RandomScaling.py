import random
from env import environment


class RandomScalingEnv(environment.DataCenterEnvironment):
    def __init__(self, env_config):
        super().__init__(env_config)
        pass



if __name__ == '__main__':
    env = environment.DataCenterEnvironment()
