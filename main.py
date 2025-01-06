from env import config, environment, loghelper
from methods import NoScaling, RandomScaling, GDCScaling
from methods import Predicter
import random
import numpy as np
import copy


seed = 1037
random.seed(seed)
np.random.seed(seed)

class TestHelper(object):
    def __init__(self, envs: list[environment.DataCenterEnvironment], agents: list[RandomScaling.RandomScalingAgent], logger: loghelper.LogHelper):
        self.agents = agents
        self.envs = envs
        self.logger = logger

    def test(self):
        total_rewards = [0 for _ in self.agents]
        dones = [False for _ in self.agents]
        states = [env.reset()[0] for env in self.envs]
        infos = [{} for _ in self.agents]

        while not all(dones):
            for i, (agent, env) in enumerate(zip(self.agents, self.envs)):
                if not dones[i]:  # Skip if this agent's environment is already done
                    action = agent.get_action(states[i])
                    next_state, reward, done, _, info = env.step(action)

                    total_rewards[i] += reward
                    states[i] = next_state
                    dones[i] = done
                    infos[i] = info

            self.logger.record(infos)

        # Log total rewards for each agent
        for agent, total_reward in zip(self.agents, total_rewards):
            print(f"Agent {agent} total_reward: {total_reward}")

        self.logger.visualize()


if __name__ == '__main__':
    env_config = config.EnvConfig()
    predicter = Predicter.SMAPredictor(env_config.ms_nums, env_config.predicter_window_size)
    envs = [environment.DataCenterEnvironment(i, env_config) for i in range(3)]
    agents = [NoScaling.NoScalingAgent(envs[0]),
              RandomScaling.RandomScalingAgent(envs[1]),
              GDCScaling.GDCScalingAgent(envs[2])]
    logger = loghelper.LogHelper(["NoScaling", "Random", "GDC"])
    test_helper = TestHelper(envs, agents, logger)
    test_helper.test()

        

