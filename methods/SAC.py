# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from datetime import datetime
from dataclasses import dataclass
import shutil, inspect
from tqdm import tqdm

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import collections
from torch.utils.tensorboard import SummaryWriter

import os, sys 
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

for _i in range(1, len(sys.argv)):
    if sys.argv[_i] == "--config" and _i + 1 < len(sys.argv):
        os.environ["AUTOSCALING_CONFIG"] = sys.argv[_i + 1]
        sys.argv.pop(_i); sys.argv.pop(_i)
        break

from env import environment
from env.configs import config_sin_smallscale, config_sin_middlescale, config_twitter_largescale, config_twitter_middlescale, config_twitter_smallscale


CONFIG = environment.CONFIG
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@dataclass
class Args:
    # Algorithm specific arguments
    epoch_steps = CONFIG.num_steps
    """ the number of steps in an epoch """
    total_timesteps: int = epoch_steps * 10000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(5e6)
    """the replay memory buffer size"""
    gamma: float = 0.93
    """the discount factor gamma"""
    tau: float = 0.01
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 512
    """the batch size of sample from the reply memory"""
    reward_shaping_record_steps: int = epoch_steps * 100
    """timestep to record reward for reward scaling"""
    learning_starts: int = epoch_steps * 500
    """timestep to start learning"""
    policy_lr: float = 5e-5
    """the learning rate of the policy network optimizer"""
    q_lr: float = 5e-4
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 2  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done) 

    def size(self): 
        return len(self.buffer)
    
def make_env(env_id, config):
    def thunk():
        env = environment.DataCenterEnvironment(env_id, config, True, "SAC")
        return env

    return thunk

def save_config(save_path):
    save_path = os.path.join(save_path, "config")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    script_path = os.path.abspath(__file__)
    config_path = inspect.getfile(CONFIG.__class__)

    shutil.copy(script_path, save_path)
    shutil.copy(config_path, save_path)


# ALGO LOGIC: initialize agent here:
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.node_nums = CONFIG.node_nums
        self.ms_nums = CONFIG.ms_nums
        self.delta = CONFIG.max_instance_update_num*2+1
        self.action_space = [self.node_nums, self.ms_nums, self.delta]
        self.feature_length_list = [self.node_nums*self.ms_nums, self.node_nums, self.node_nums, self.ms_nums*CONFIG.history_lamda_length, CONFIG.history_step_length]

        # action rescaling, to standardize the action space here.
        self.register_buffer(
            "action_scale",
            torch.tensor(
                torch.tensor([self.node_nums, self.ms_nums, self.delta]),
                dtype=torch.float32,
            ),
        )

        self.dnn = nn.Sequential(
            layer_init(nn.Linear(np.sum(self.feature_length_list) + len(self.action_space), 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            # layer_init(nn.Linear(512, 512)),
            # nn.ReLU(),
            # layer_init(nn.Linear(512, 512)),
            # nn.ReLU(),
            # layer_init(nn.Linear(512, 512)),
            # nn.ReLU(),
            # layer_init(nn.Linear(512, 512)),
            # nn.ReLU(),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(512, 1)),
        )

    def _standardize_state(self, ob) -> torch.Tensor:
        """ 标准化状态，支持批次形状，同时展平给DNN输入 """
        batch_size = ob.shape[0]
        fl = self.feature_length_list
        total_features = fl[0] + fl[1] + fl[2] + fl[3] + fl[4]
        res = torch.zeros((batch_size, total_features), dtype=torch.float32, device=device)

        deploy_norm = min(
            CONFIG.node_max_cpu_resource / CONFIG.ms_max_cpu_resource,
            CONFIG.node_min_memory_resource / CONFIG.ms_min_memory_resource
        )
        res[:, :fl[0]] = ob[:, 0].view(batch_size, -1) / deploy_norm
        res[:, fl[0]:fl[0]+fl[1]] = ob[:, 1, 0] / CONFIG.node_max_cpu_resource
        res[:, fl[0]+fl[1]:fl[0]+fl[1]+fl[2]] = ob[:, 2, 0] / CONFIG.node_max_memory_resource
        # res[:, fl[0]+fl[1]+fl[2]:fl[0]+fl[1]+fl[2]+fl[3]] = (ob[:, 3, :, 0] / CONFIG.estimated_max_lamda)

        # history lamda: 原 for i in range(H) 逐块赋值; 等价于把 ob[:,4]/ob[:,5] 按 node 轴拼起来取前 H 列,
        # 再转置展平 (block i == ob[:,4/5][:,:,i])。逐位相同, 但只需 1~2 次 GPU op。
        H = CONFIG.history_lamda_length
        lamda_src = torch.cat([ob[:, 4], ob[:, 5]], dim=2)[:, :, :H]   # (batch, ms_nums, H)
        res[:, fl[0]+fl[1]+fl[2]:fl[0]+fl[1]+fl[2]+fl[3]] = \
            lamda_src.permute(0, 2, 1).reshape(batch_size, -1) / CONFIG.estimated_max_lamda

        # history step one-hot: 原 for i in range(Hs) 逐标量赋值; 等价于一次性 advanced indexing。
        Hs = CONFIG.history_step_length
        idx = torch.arange(Hs, device=ob.device)
        rows = idx // self.node_nums
        cols = idx % self.ms_nums
        res[:, fl[0]+fl[1]+fl[2]+fl[3]:fl[0]+fl[1]+fl[2]+fl[3]+fl[4]] = ob[:, 6, rows, cols]

        return res
    
    def _standardize_action(self, a) -> torch.Tensor:
        """ 标准化动作，支持批次形状 """
        return a / self.action_scale    # 逐元素点除

    def forward(self, ob, a):
        x = self._standardize_state(ob) # 标准化从环境观测到的状态
        a = self._standardize_action(a)
        x = torch.cat([x, a], 1)
        x = self.dnn(x)
        x = self.critic(x)
        return x
    
    def save(self, path, name):
        save_path = os.path.join(path, name)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        
        torch.save(self.state_dict(), save_path)

    def load(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env=None):
        super().__init__()
        self.node_nums = CONFIG.node_nums
        self.ms_nums = CONFIG.ms_nums
        self.delta = CONFIG.max_instance_update_num*2+1
        self.action_space = [self.node_nums, self.ms_nums, self.delta]
        self.feature_length_list = [self.node_nums*self.ms_nums, self.node_nums, self.node_nums, self.ms_nums*CONFIG.history_lamda_length, CONFIG.history_step_length]
        
        # 特征提取层
        self.dnn = nn.Sequential(
            layer_init(nn.Linear(np.sum(self.feature_length_list), 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            # layer_init(nn.Linear(512, 512)),
            # nn.ReLU(),
            # layer_init(nn.Linear(512, 512)),
            # nn.ReLU(),
            # layer_init(nn.Linear(512, 512)),
            # nn.ReLU(),
            # layer_init(nn.Linear(512, 512)),
            # nn.ReLU(),
        )

        # 输出层
        self.fc_mean = layer_init(nn.Linear(512, len(self.action_space)), std=0.01)
        self.fc_logstd = layer_init(nn.Linear(512, len(self.action_space)), std=0.01)

        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                torch.tensor([self.node_nums, self.ms_nums, self.delta]) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                torch.tensor([self.node_nums, self.ms_nums, self.delta]) / 2.0,
                dtype=torch.float32,
            ),
        )

    def _standardize_state(self, ob) -> torch.Tensor:
        """ 标准化状态，支持批次形状，同时展平给DNN输入 """
        batch_size = ob.shape[0]
        fl = self.feature_length_list
        total_features = fl[0] + fl[1] + fl[2] + fl[3] + fl[4]
        res = torch.zeros((batch_size, total_features), dtype=torch.float32, device=device)

        deploy_norm = min(
            CONFIG.node_max_cpu_resource / CONFIG.ms_max_cpu_resource,
            CONFIG.node_min_memory_resource / CONFIG.ms_min_memory_resource
        )
        res[:, :fl[0]] = ob[:, 0].view(batch_size, -1) / deploy_norm
        res[:, fl[0]:fl[0]+fl[1]] = ob[:, 1, 0] / CONFIG.node_max_cpu_resource
        res[:, fl[0]+fl[1]:fl[0]+fl[1]+fl[2]] = ob[:, 2, 0] / CONFIG.node_max_memory_resource
        # res[:, fl[0]+fl[1]+fl[2]:fl[0]+fl[1]+fl[2]+fl[3]] = (ob[:, 3, :, 0] / CONFIG.estimated_max_lamda)

        # history lamda: 原 for i in range(H) 逐块赋值; 等价于把 ob[:,4]/ob[:,5] 按 node 轴拼起来取前 H 列,
        # 再转置展平 (block i == ob[:,4/5][:,:,i])。逐位相同, 但只需 1~2 次 GPU op。
        H = CONFIG.history_lamda_length
        lamda_src = torch.cat([ob[:, 4], ob[:, 5]], dim=2)[:, :, :H]   # (batch, ms_nums, H)
        res[:, fl[0]+fl[1]+fl[2]:fl[0]+fl[1]+fl[2]+fl[3]] = \
            lamda_src.permute(0, 2, 1).reshape(batch_size, -1) / CONFIG.estimated_max_lamda

        # history step one-hot: 原 for i in range(Hs) 逐标量赋值; 等价于一次性 advanced indexing。
        Hs = CONFIG.history_step_length
        idx = torch.arange(Hs, device=ob.device)
        rows = idx // self.node_nums
        cols = idx % self.ms_nums
        res[:, fl[0]+fl[1]+fl[2]+fl[3]:fl[0]+fl[1]+fl[2]+fl[3]+fl[4]] = ob[:, 6, rows, cols]

        return res
    
    def forward(self, ob):
        x = self._standardize_state(ob) # 标准化从环境观测到的状态
        x = self.dnn(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
    
    def save(self, path, name):
        save_path = os.path.join(path, name)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        
        torch.save(self.state_dict(), save_path)

    def load(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))

class SACAgent(object):
    def __init__(self, config):
        self.config = config
        self.actor = Actor().to(device)

    def load(self, path):
        load_path = os.path.join(path, "model.pth")
        self.actor.load_state_dict(torch.load(load_path, weights_only=True, map_location=torch.device('cpu')))

    def get_action(self, x):
        """
        predict, 供test对比实验调用，不在训练中被调用
        """
        # self.actorcrtic.eval()
        x = torch.Tensor(x).unsqueeze(0)
        action, _, _ = self.actor.get_action(x)
        action = action.cpu().detach().numpy()[0]
        return action

class RewardScaler:
    def __init__(self, record_steps, epsilon=1e-8):
        self.buffer = []
        self.mean = 0
        self.var = 0
        self.count = 0
        self.record_steps = record_steps
        self.epsilon = epsilon

    def record(self, reward):
        """ 记录奖励 """
        self.buffer.append(reward)
        self.count += 1

        # 更新均值和方差
        if self.count >= self.record_steps:
            self.mean = np.mean(self.buffer)
            self.var = np.var(self.buffer)

    def reward_shaping(self, reward):
        """ 计算奖励。通过前若干轮的探索轮次计算均值和方差，然后归一化 """
        new_reward = (reward - self.mean) / (self.var ** 0.5 + self.epsilon)
        return float(new_reward)
    
def seed_all(seed):
    # TRY NOT TO MODIFY: seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train():
    args = tyro.cli(Args)

    save_path = os.path.join(CONFIG.model_path, CONFIG.config_name, datetime.now().strftime("%m%d"), datetime.now().strftime("%H%M"), "SAC")
    writer = SummaryWriter(save_path)
    save_config(save_path)

    # env setup
    envs = gym.vector.AsyncVectorEnv(
        [make_env(i, CONFIG) for i in range(args.num_envs)],
    )
    # envs = make_env(0, CONFIG)()
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # reward scaling
    reward_scaler = RewardScaler(record_steps=args.reward_shaping_record_steps)

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    # continue training
    # actor.load("model/sin_largescale/0314/1637/SAC/model.pth")
    # qf1.load("model/sin_largescale/0314/1637/SAC/qf1.pth")
    # qf2.load("model/sin_largescale/0314/1637/SAC/qf2.pth")
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(int(args.buffer_size))
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=CONFIG.seed)
    tqdm_records = []
    total_reward = []
    total_y = []
    total_Qt = []
    total_delay = {"t_all": [], "t_exe": [], "t_route": []}
    total_vload = []
    total_ns = []
    total_cost = []
    total_s_cost = []
    total_d_cost = []
    total_node_using_num = []
    total_image_nums = []
    total_rsr = []  # 请求成功率
    total_penalty = []
    with tqdm(total=args.total_timesteps / args.epoch_steps, desc="Training", unit="it") as pbar:
        for global_step in range(args.total_timesteps):
            # ALGO LOGIC: put action logic here
            if global_step < args.learning_starts:
                actions = np.array([[random.uniform(0, CONFIG.node_nums), random.uniform(0, CONFIG.ms_nums), random.uniform(0, CONFIG.max_instance_update_num)] for _ in range(envs.num_envs)])
            else:
                actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
                actions = actions.detach().cpu().numpy()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)

            # reward shaping
            if global_step < args.reward_shaping_record_steps:
                reward_scaler.record(rewards)
                continue

            rewards = reward_scaler.reward_shaping(rewards)
            
            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]
            rb.add(obs, actions, rewards, real_next_obs, terminations)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # ALGO LOGIC: training.
            if global_step > args.learning_starts + args.reward_shaping_record_steps:
                data = {"observations": None, "actions": None, "rewards": None, "next_observations": None, "dones": None}
                data["observations"], data["actions"], data["rewards"], data["next_observations"], data["dones"] = rb.sample(args.batch_size)

                # flatten the batch
                data["observations"] = data["observations"].reshape((-1,) + envs.single_observation_space.shape)
                data["actions"] = data["actions"].reshape((-1,) + (envs.single_action_space.shape))
                data["rewards"] = data["rewards"].reshape(-1)
                data["next_observations"] = data["next_observations"].reshape((-1,) + envs.single_observation_space.shape)
                data["dones"] = data["dones"].reshape(-1)

                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = actor.get_action(torch.Tensor(data["next_observations"]).to(device))
                    qf1_next_target = qf1_target(torch.Tensor(data["next_observations"]).to(device), next_state_actions)
                    qf2_next_target = qf2_target(torch.Tensor(data["next_observations"]).to(device), next_state_actions)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    next_q_value = torch.Tensor(data["rewards"]).to(device).flatten() + (1 - torch.Tensor(data["dones"]).to(device).flatten()) * args.gamma * (min_qf_next_target).view(-1)

                qf1_a_values = qf1(torch.Tensor(data["observations"]).to(device), torch.Tensor(data["actions"]).to(device)).view(-1)
                qf2_a_values = qf2(torch.Tensor(data["observations"]).to(device), torch.Tensor(data["actions"]).to(device)).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                # optimize the model
                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                    for _ in range(
                        args.policy_frequency
                    ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                        pi, log_pi, _ = actor.get_action(torch.Tensor(data["observations"]).to(device))
                        qf1_pi = qf1(torch.Tensor(data["observations"]).to(device), pi)
                        qf2_pi = qf2(torch.Tensor(data["observations"]).to(device), pi)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi)
                        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        actor_optimizer.step()

                        if args.autotune:
                            with torch.no_grad():
                                _, log_pi, _ = actor.get_action(torch.Tensor(data["observations"]).to(device))
                            alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                            a_optimizer.zero_grad()
                            alpha_loss.backward()
                            a_optimizer.step()
                            alpha = log_alpha.exp().item()

                # update the target networks
                if global_step % args.target_network_frequency == 0:
                    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                
                if global_step % CONFIG.num_steps == 0:
                    # running data
                    writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                    writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                    writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                    writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                    writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                    writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                    writer.add_scalar("losses/alpha", alpha, global_step)
                    
                    writer.add_scalar(
                        "charts/SPS",
                        int(global_step / (time.time() - start_time)),
                        global_step,
                    )
                    if args.autotune:
                        writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

            # logging
            if infos != {}:
                total_reward.append(np.mean(rewards))
                total_y.append(np.mean(infos['y']))
                total_Qt.append(np.mean(infos['Qt']))
                total_delay["t_all"].append(np.mean(infos['t_all']))
                total_delay["t_exe"].append(np.mean(infos['t_exe']))
                total_delay["t_route"].append(np.mean(infos['t_route']))
                total_vload.append(np.mean(infos['vload']))
                total_ns.append(np.mean(infos['ns']))
                total_cost.append(np.mean(infos['cost']))
                total_s_cost.append(np.mean(infos['static_cost']))
                total_d_cost.append(np.mean(infos['dynamic_cost']))
                total_node_using_num.append(np.mean(infos['node_using_num']))
                total_image_nums.append(np.mean(infos['image_nums']))
                total_rsr.append(np.mean(infos['request_success_rate']))
                total_penalty.append(np.mean(infos['penalty']))

            if global_step % CONFIG.num_steps == 0:
                iteration = global_step // CONFIG.num_steps
                actor.save(save_path, "model.pth")
                qf1.save(save_path, "qf1.pth")
                qf2.save(save_path, "qf2.pth")
                if iteration % 5000 == 0:
                    actor.save(save_path, f"model_{iteration}.pth")
                    qf1.save(save_path, f"qf1_{iteration}.pth")
                    qf2.save(save_path, f"qf2_{iteration}.pth")

                # result data
                # print(f"Iteration: {iteration}, Total Reward: {np.sum(total_reward)}")
                writer.add_scalar("charts/reward", np.sum(total_reward), iteration)
                writer.add_scalar("charts/y", np.mean(total_y), iteration)
                writer.add_scalar("charts/Qt", np.mean(total_Qt), iteration)
                writer.add_scalar("charts/t_all", np.mean(total_delay["t_all"]), iteration)
                writer.add_scalar("charts/t_exe", np.mean(total_delay["t_exe"]), iteration)
                writer.add_scalar("charts/t_route", np.mean(total_delay["t_route"]), iteration)
                writer.add_scalar("charts/vload", np.mean(total_vload), iteration)
                writer.add_scalar("charts/ns", np.sum(total_ns), iteration)
                writer.add_scalar("charts/cost", np.mean(total_cost), iteration)
                writer.add_scalar("charts/s_cost", np.mean(total_s_cost), iteration)
                writer.add_scalar("charts/d_cost", np.mean(total_d_cost), iteration)
                writer.add_scalar("charts/node_using_num", np.mean(total_node_using_num), iteration)
                writer.add_scalar("charts/image_nums", np.mean(total_image_nums), iteration)
                writer.add_scalar("charts/rsr", np.mean(total_rsr), iteration)
                writer.add_scalar("charts/penalty", np.sum(total_penalty), iteration)
                
                # tqdm update
                tqdm_records.append(np.sum(total_reward))
                pbar.set_postfix(reward=np.mean(tqdm_records[-100:]))
                pbar.update(1)

                # reset data
                total_reward.clear()
                total_y.clear()
                total_Qt.clear()
                total_delay["t_all"].clear(), total_delay["t_exe"].clear(), total_delay["t_route"].clear()
                total_vload.clear()
                total_ns.clear()
                total_cost.clear()
                total_s_cost.clear()
                total_d_cost.clear()
                total_node_using_num.clear()
                total_image_nums.clear()
                total_rsr.clear()
                total_penalty.clear()

    envs.close()
    writer.close()

if __name__ == "__main__":
    seed_all(CONFIG.seed)
    train()
