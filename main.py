from env.configs import config_sin_smallscale, config_sin_middlescale, config_twitter_largescale, config_twitter_middlescale, config_twitter_smallscale
from env import environment, loghelper
from methods import NoScaling, RandomScaling, GDCScaling, PPO_dnn, ProScaling, PPO_dnn_las
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
    env_config = config_twitter_smallscale.EnvConfig()
    predicter = Predicter.SMAPredictor(env_config.ms_nums, env_config.predicter_window_size)

    # envs = [environment.DataCenterEnvironment(i, env_config) for i in range(5)]
    # ppoAgent1 = PPO_dnn.PPOAgent(env_config)
    # ppoAgent1.load("model/twitter_smallscale_v2/0116/1017/PPO_dnn")
    # ppoAgent2 = PPO_dnn.PPOAgent(env_config)
    # ppoAgent2.load("model/twitter_smallscale_v2/0116/1037/PPO_dnn")
    # agents = [NoScaling.NoScalingAgent(envs[0]),
    #           GDCScaling.GDCScalingAgent(envs[1]),
    #           ProScaling.ProScalingAgent(envs[2]),
    #           ppoAgent1,
    #           ppoAgent2]
    # logger = loghelper.LogHelper(["NoScaling", "GDC", "ProScaling", "DRL1", "DRL2"], envs)

    # envs = [environment.DataCenterEnvironment(i, env_config) for i in range(3)]
    # ppoAgent2 = PPO_dnn.PPOAgent(envs[2], env_config)
    # ppoAgent2.load("model/0109/175325/PPO_dnn")
    # agents = [GDCScaling.GDCScalingAgent(envs[0]),
    #           GDCScaling.GDCScalingAgent_Ideal(envs[1]),
    #           ppoAgent2]
    # logger = loghelper.LogHelper(["GDC", "Ideal", "DRL"])

    # envs = [environment.DataCenterEnvironment(i, env_config) for i in range(4)]
    # agents = [GDCScaling.GDCScalingAgent(envs[0]),
    #           GDCScaling.GDCScalingAgent_Ideal(envs[1]),
    #           NoScaling.NoScalingAgent(envs[2]),
    #           ProScaling.ProScalingAgent(envs[3])]
    # logger = loghelper.LogHelper(["GDC", "Ideal", "NS", "Proscale"], envs)

    # envs = [environment.DataCenterEnvironment(i, env_config) for i in range(5)]
    # ppoAgent2 = PPO_dnn.PPOAgent(envs[4], env_config)
    # ppoAgent2.load("model/twitter_largescale/0112/2142/PPO_dnn")
    # agents = [GDCScaling.GDCScalingAgent(envs[0]),
    #           GDCScaling.GDCScalingAgent_Ideal(envs[1]),
    #           NoScaling.NoScalingAgent(envs[2]),
    #           ProScaling.ProScalingAgent(envs[3]),
    #           ppoAgent2]
    # logger = loghelper.LogHelper(["GDC", "Ideal", "NoScaling", "ProScaling", "DRL"], envs)

    envs = [environment.DataCenterEnvironment(i, env_config) for i in range(5)]
    ppoAgent2 = PPO_dnn.PPOAgent(env_config)
    ppoAgent2.load("model/twitter_smallscale_v2/0116/1037/PPO_dnn")
    agents = [NoScaling.NoScalingAgent(envs[0]),
              RandomScaling.RandomScalingAgent(envs[1]),
              ProScaling.ProScalingAgent(envs[2]),
              GDCScaling.GDCScalingAgent_Ideal(envs[3]),
              ppoAgent2,
              ]
    logger = loghelper.LogHelper(["NoScaling", "Random", "ProScaling", "Ideal", "DRL"], envs)

    # envs = [environment.DataCenterEnvironment(i, env_config) for i in range(4)]
    # ppoAgent1 = PPO_dnn.PPOAgent(envs[2], env_config)
    # ppoAgent1.load("model/twitter_smallscale_v2/0114/1525/PPO_dnn")
    # ppoAgent2 = PPO_dnn_las.PPOLasAgent(envs[3], env_config)
    # ppoAgent2.load("model/twitter_smallscale_v2/0114/1537/PPO_dnn_las")
    # agents = [NoScaling.NoScalingAgent(envs[0]),
    #           ProScaling.ProScalingAgent(envs[1]),
    #           ppoAgent1,
    #           ppoAgent2]
    # logger = loghelper.LogHelper(["NoScaling", "ProScaling", "DRL", "DRL_las"], envs)

    test_helper = TestHelper(envs, agents, logger)
    test_helper.test()