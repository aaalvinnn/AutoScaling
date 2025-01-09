from env import config, environment, loghelper
from methods import NoScaling, RandomScaling, GDCScaling, PPO_cnn, PPO_dnn, PPO_dnn_v2
from methods import Predicter
import random
import numpy as np
from tqdm import tqdm


seed = 1037
random.seed(seed)
np.random.seed(seed)

class TestHelper(object):
    def __init__(self, envs: list[environment.DataCenterEnvironment], agents: list[RandomScaling.RandomScalingAgent], logger: loghelper.LogHelper):
        self.agents = agents
        self.envs = envs
        self.logger = logger

    def test(self, total_steps=288):
        total_rewards = [0 for _ in self.agents]
        dones = [False for _ in self.agents]
        states = [env.reset()[0] for env in self.envs]
        infos = [{} for _ in self.agents]

        with tqdm(total=total_steps, desc="Epoch Progress", unit="step") as pbar:
            while not all(dones):
                for i, (agent, env) in enumerate(zip(self.agents, self.envs)):
                    if not dones[i]:  # Skip if this agent's environment is already done
                        action = agent.get_action(states[i])
                        next_state, reward, done, _, info = env.step(action)

                        total_rewards[i] += reward
                        states[i] = next_state
                        dones[i] = done
                        infos[i] = info

                # update progress bar
                pbar.update(1)

                self.logger.record(infos)

            # Log total rewards for each agent
            for agent, total_reward in zip(self.logger.agents_name, total_rewards):
                print(f"Agent {agent} total_reward: {total_reward}")

        self.logger.visualize()
        self.logger.save_data()


if __name__ == '__main__':
    env_config = config.EnvConfig()
    predicter = Predicter.SMAPredictor(env_config.ms_nums, env_config.predicter_window_size)

    # envs = [environment.DataCenterEnvironment(i, env_config) for i in range(5)]
    # ppoAgent1 = PPO_dnn_v2.PPOAgent(envs[2], env_config)
    # ppoAgent1.load("model/0109/124836/PPO_dnn_v2")
    # ppoAgent2 = PPO_dnn.PPOAgent(envs[3], env_config)
    # ppoAgent2.load("model/0109/124827/PPO_dnn")
    # agents = [NoScaling.NoScalingAgent(envs[0]),
    #           GDCScaling.GDCScalingAgent(envs[1]),
    #           GDCScaling.GDCScalingAgent_Ideal(envs[2]),
    #           ppoAgent1,
    #           ppoAgent2]
    # logger = loghelper.LogHelper(["NoScaling", "GDC", "Ideal", "DRL2", "DRL1"])

    # envs = [environment.DataCenterEnvironment(i, env_config) for i in range(3)]
    # ppoAgent2 = PPO_dnn.PPOAgent(envs[2], env_config)
    # ppoAgent2.load("model/0109/175325/PPO_dnn")
    # agents = [GDCScaling.GDCScalingAgent(envs[0]),
    #           GDCScaling.GDCScalingAgent_Ideal(envs[1]),
    #           ppoAgent2]
    # logger = loghelper.LogHelper(["GDC", "Ideal", "DRL"])

    # envs = [environment.DataCenterEnvironment(i, env_config) for i in range(3)]
    # agents = [GDCScaling.GDCScalingAgent(envs[0]),
    #           GDCScaling.GDCScalingAgent_Ideal(envs[1]),
    #           NoScaling.NoScalingAgent(envs[2])]
    # logger = loghelper.LogHelper(["GDC", "Ideal", "NS"])

    envs = [environment.DataCenterEnvironment(i, env_config) for i in range(4)]
    ppoAgent2 = PPO_dnn_v2.PPOAgent(envs[3], env_config)
    ppoAgent2.load("model/0109/175327/PPO_dnn_v2")
    agents = [GDCScaling.GDCScalingAgent(envs[0]),
              GDCScaling.GDCScalingAgent_Ideal(envs[1]),
              NoScaling.NoScalingAgent(envs[2]),
              ppoAgent2]
    logger = loghelper.LogHelper(["GDC", "Ideal", "NoScaling", "DRL"])

    test_helper = TestHelper(envs, agents, logger)
    test_helper.test()