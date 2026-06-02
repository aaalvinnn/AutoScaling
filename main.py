from env.configs import config_sin_smallscale, config_sin_middlescale, config_twitter_largescale, config_twitter_middlescale, config_twitter_smallscale
from env import environment, loghelper
from methods import NoScaling, RandomScaling, GDCScaling, PPO_dnn, ProScaling, SAC, HPA
from methods import Predicter
import random
import numpy as np
from tqdm import tqdm

# seed = 1037
# random.seed(seed)
# np.random.seed(seed)

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

        return total_rewards, self.logger.get_data()

    def smooth_test(self, running_times=10):
        rewards = []
        infos = []
        ave_rewards = []
        ave_infos = []
        
        with tqdm(total=running_times, desc="Running Progress", unit="Epoch") as pbar:
            for i in range(running_times):
                PPO_dnn.seed_all(i)
                reward, info = self.test()
                rewards.append(reward)
                infos.append(info)
                self.logger.clear()

                pbar.update(1)

        for i, agent in enumerate(self.logger.agents_name):
            ave_rewards.append(np.mean(rewards[i], axis=0))
            for key in infos[i]:
                _data = 0
                for j in range(running_times):
                    _data += infos[j][agent][key]
                ave_infos.append({key: np.mean(_data, axis=0)})

        self.logger.replace_data(ave_infos)
        self.logger.visualize()
        self.logger.save_data()


LGDRL_MODEL_PATH = {
    "sin_smallscale": "trained_models/sin_smallscale/0202/1159/PPO_dnn",
    "sin_middlescale": "trained_models/sin_middlescale/0202/0758_best/PPO_dnn",
    # "sin_largescale": "trained_models/sin_largescale/0205/1249/PPO_dnn",
    # "sin_largescale": "model/sin_largescale/0428/1936_V=10_1/PPO_dnn",
    "sin_largescale": "model/sin_largescale/0531/1359/PPO_dnn",
    "twitter_smallscale": "trained_models/twitter_smallscale/0205/2232/PPO_dnn",
    "twitter_middlescale": "trained_models/twitter_middlescale/0202/1555/PPO_dnn",
    # "twitter_largescale": "trained_models/twitter_largescale/0203/2143_best/PPO_dnn",
    # "twitter_largescale": "trained_models/twitter_largescale/0412_V0_3"
    # "twitter_largescale": "params_exp/V/N_change=3/10/1",
    "twitter_largescale": "model/twitter_largescale/0530/1829/PPO_dnn"
}

RLAGENT_MODEL_PATH = {
    "sin_smallscale": "trained_models/sin_smallscale/0314/0953_best/SAC",
    "sin_middlescale": "trained_models/sin_middlescale/0312/1307_best/SAC",
    # "sin_largescale": "trained_models/sin_largescale/0314/1637/SAC",
    "sin_largescale": "model/sin_largescale/0529/2158/SAC",
    "twitter_smallscale": "trained_models/twitter_smallscale/0312/1308_best/SAC",
    "twitter_middlescale": "trained_models/twitter_middlescale/0315/1644/SAC",
    # "twitter_largescale": "trained_models/twitter_largescale/0313/1046/SAC"
    "twitter_largescale": "model/twitter_largescale/0603/1507/SAC"
}

if __name__ == '__main__':
    env_config = environment.CONFIG
    predicter = Predicter.SMAPredictor(env_config.ms_nums, env_config.predicter_window_size)

    # envs = [environment.DataCenterEnvironment(i, env_config) for i in range(4)]
    # ppoAgent1 = PPO_dnn.PPOAgent(config=env_config)
    # ppoAgent1.load(LGDRL_MODEL_PATH[env_config.config_name])
    # ppoAgent2 = SAC.SACAgent(config=env_config)
    # ppoAgent2.load(RLAGENT_MODEL_PATH[env_config.config_name])
    # agents = [HPA.HPA(envs[0]),
    #           ProScaling.ProScalingAgent(envs[1]),
    #           ppoAgent2,
    #           ppoAgent1]
    # logger = loghelper.LogHelper(["HPA", "Proscale", "RL Agent", "LGDRL"], envs)

    # envs = [environment.DataCenterEnvironment(i, env_config) for i in range(1)]
    # ppoAgent2 = SAC.SACAgent(config=env_config)
    # ppoAgent2.load(RLAGENT_MODEL_PATH[env_config.config_name])
    # agents = [ppoAgent2]
    # logger = loghelper.LogHelper(["RL Agent"], envs)

    # envs = [environment.DataCenterEnvironment(i, env_config) for i in range(3)]
    # agents = [HPA.HPA(envs[0]),
    #           ProScaling.ProScalingAgent(envs[1]),
    #           GDCScaling.GDCScalingAgent_Ideal(envs[2])
    #           ]
    # logger = loghelper.LogHelper(["HPA", "Proscale", "HPA_ideal"], envs)

    # envs = [environment.DataCenterEnvironment(i, env_config) for i in range(5)]
    # ppoAgent1 = PPO_dnn.PPOAgent(config=env_config)
    # ppoAgent1.load("trained_models/params_exp/V/0604/V=1/PPO_dnn/model_dnn_5000.pth")
    # ppoAgent2 = PPO_dnn.PPOAgent(config=env_config)
    # ppoAgent2.load("trained_models/params_exp/V/0604/V=2/PPO_dnn/model_dnn.pth")
    # ppoAgent3 = PPO_dnn.PPOAgent(config=env_config)
    # ppoAgent3.load("model/twitter_largescale/0530/1829/PPO_dnn/model_dnn_best.pth")
    # ppoAgent4 = PPO_dnn.PPOAgent(config=env_config)
    # ppoAgent4.load("trained_models/params_exp/V/0604/V=100/PPO_dnn/model_dnn.pth")
    # ppoAgent5 = PPO_dnn.PPOAgent(config=env_config)
    # ppoAgent5.load("trained_models/params_exp/V/0604/V=1000/PPO_dnn/model_dnn.pth")
    # agents = [ppoAgent1,
    #           ppoAgent2,
    #           ppoAgent3,
    #           ppoAgent4,
    #           ppoAgent5]
    # logger = loghelper.LogHelper(["V=1", "V=2", "V=10", "V=100", "V=1000"], envs)

    # envs = [environment.DataCenterEnvironment(i, env_config) for i in range(3)]
    # ppoAgent1 = PPO_dnn.PPOAgent(config=env_config, delta=1)
    # ppoAgent1.load("trained_models/params_exp/delta/0610/delta=1/PPO_dnn/model_dnn.pth")
    # ppoAgent2 = PPO_dnn.PPOAgent(config=env_config, delta=2)
    # ppoAgent2.load("trained_models/params_exp/delta/0610/delta=2/PPO_dnn/model_dnn.pth")
    # ppoAgent3 = PPO_dnn.PPOAgent(config=env_config, delta=3)
    # ppoAgent3.load("model/twitter_largescale/0530/1829/PPO_dnn/model_dnn.pth")
    # agents = [ppoAgent1,
    #           ppoAgent2,
    #           ppoAgent3]
    # logger = loghelper.LogHelper(["delta=1", "delta=2", "delta=3"], envs)

    envs = [environment.DataCenterEnvironment(i, env_config) for i in range(1)]
    ppoAgent1 = PPO_dnn.PPOAgent(config=env_config)
    ppoAgent1.load("model/twitter_largescale/0530/1829/PPO_dnn/model_dnn.pth")
    agents = [ppoAgent1]
    logger = loghelper.LogHelper(["V=10"], envs)

    test_helper = TestHelper(envs, agents, logger)
    test_helper.test()
    # test_helper.smooth_test(running_times=3)