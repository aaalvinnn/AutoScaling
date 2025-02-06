from env.configs import config_sin_smallscale, config_sin_middlescale, config_twitter_largescale, config_twitter_middlescale, config_twitter_smallscale
from env import environment, loghelper
from methods import NoScaling, RandomScaling, GDCScaling, PPO_dnn, ProScaling
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
                        # # debug
                        # if env.timeslot.get_now() in (180, 182, 184):
                        #     action = (random.randint(0, 4), random.randint(0, 4), 0)
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
    env_config = environment.CONFIG
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

    # envs = [environment.DataCenterEnvironment(i, env_config) for i in range(5)]
    # agents = [GDCScaling.GDCScalingAgent(envs[0]),
    #           GDCScaling.GDCScalingAgent_Ideal(envs[1]),
    #           NoScaling.NoScalingAgent(envs[2]),
    #           ProScaling.ProScalingAgent(envs[3]),
    #           RandomScaling.RandomScalingAgent(envs[4])]
    # logger = loghelper.LogHelper(["GDC", "Ideal", "NS", "Proscale", "Random"], envs)

    # envs = [environment.DataCenterEnvironment(i, env_config) for i in range(5)]
    # ppoAgent2 = PPO_dnn.PPOAgent(envs[4], env_config)
    # ppoAgent2.load("model/twitter_largescale/0112/2142/PPO_dnn")
    # agents = [GDCScaling.GDCScalingAgent(envs[0]),
    #           GDCScaling.GDCScalingAgent_Ideal(envs[1]),
    #           NoScaling.NoScalingAgent(envs[2]),
    #           ProScaling.ProScalingAgent(envs[3]),
    #           ppoAgent2]
    # logger = loghelper.LogHelper(["GDC", "Ideal", "NoScaling", "ProScaling", "DRL"], envs)

    # envs = [environment.DataCenterEnvironment(i, env_config) for i in range(6)]
    # ppoAgent2 = PPO_dnn.PPOAgent(env_config)
    # ppoAgent2.load("model/sin_largescale/0130/0015/PPO_dnn")
    # agents = [NoScaling.NoScalingAgent(envs[0]),
    #           RandomScaling.RandomScalingAgent(envs[1]),
    #           ProScaling.ProScalingAgent(envs[2]),
    #           GDCScaling.GDCScalingAgent_Ideal(envs[3]),
    #           GDCScaling.GDCScalingAgent(envs[4]),
    #           ppoAgent2,
    #           ]
    # logger = loghelper.LogHelper(["NoScaling", "Random", "ProScaling", "Ideal", "GDC", "DRL"], envs)

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

    # envs = [environment.DataCenterEnvironment(i, env_config) for i in range(4)]
    # ppoAgent2 = PPO_dnn.PPOAgent(env_config)
    # ppoAgent2.load("model/sin_middlescale/0202/0758/PPO_dnn")
    # agents = [GDCScaling.GDCScalingAgent_Ideal(envs[0]),
    #           GDCScaling.GDCScalingAgent(envs[1]),
    #           ProScaling.ProScalingAgent(envs[2]),
    #           ppoAgent2,
    #           ]
    # logger = loghelper.LogHelper(["Ideal", "GDC", "ProScaling", "DRL"], envs)

    # envs = [environment.DataCenterEnvironment(i, env_config) for i in range(1)]
    # ppoAgent2 = PPO_dnn.PPOAgent(env_config)
    # ppoAgent2.load("model/sin_smallscale/0128/2012_good/PPO_dnn")
    # agents = [ppoAgent2]
    # logger = loghelper.LogHelper(["DRL"], envs)

    envs = [environment.DataCenterEnvironment(i, env_config) for i in range(3)]
    ppoAgent2 = PPO_dnn.PPOAgent(env_config)
    ppoAgent2.load("model/twitter_smallscale/0205/2232/PPO_dnn")
    agents = [RandomScaling.RandomScalingAgent(envs[0]),
              ProScaling.ProScalingAgent(envs[1]),
              ppoAgent2]
    logger = loghelper.LogHelper(["Random", "ProScaling", "DRL"], envs)

    test_helper = TestHelper(envs, agents, logger)
    test_helper.test()