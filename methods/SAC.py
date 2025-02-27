# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from datetime import datetime
from dataclasses import dataclass
import shutil, inspect

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
from env import environment
from env.configs import config_sin_smallscale, config_sin_middlescale, config_twitter_largescale, config_twitter_middlescale, config_twitter_smallscale


CONFIG = environment.CONFIG


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    replay_buffer_size: int = 5e4
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e2
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
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
        return np.array(state), action, reward, np.array(next_state), done 

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
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.node_nums = CONFIG.node_nums
        self.ms_nums = CONFIG.ms_nums
        self.delta = CONFIG.max_instance_update_num*2+1
        self.feature_length_list = [self.node_nums*self.ms_nums, self.node_nums, self.node_nums, self.ms_nums*CONFIG.history_lamda_length, CONFIG.history_step_length]
        
        self.fc1 = nn.Linear(
            np.array(np.sum(self.feature_length_list) + np.prod(env.single_action_space.shape)),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def _standardize_state(self, ob) -> torch.Tensor:
        """ 标准化状态，支持批次形状，同时展平给DNN输入 """
        batch_size = ob.shape[0]
        total_features = np.sum(self.feature_length_list)
        fl = self.feature_length_list
        res = torch.zeros((batch_size, total_features), dtype=torch.float32, device=CONFIG.device)

        res[:, :fl[0]] = ob[:, 0].view(batch_size, -1) / min(
            CONFIG.node_max_cpu_resource / CONFIG.ms_max_cpu_resource,
            CONFIG.node_min_memory_resource / CONFIG.ms_min_memory_resource
        )
        res[:, fl[0]:fl[0]+fl[1]] = ob[:, 1, 0] / CONFIG.node_max_cpu_resource
        res[:, fl[0]+fl[1]:fl[0]+fl[1]+fl[2]] = ob[:, 2, 0] / CONFIG.node_max_memory_resource
        # res[:, fl[0]+fl[1]+fl[2]:fl[0]+fl[1]+fl[2]+fl[3]] = (ob[:, 3, :, 0] / CONFIG.estimated_max_lamda)
        for i in range(CONFIG.history_lamda_length):
            l = fl[0]+fl[1]+fl[2] + self.ms_nums*i
            r = fl[0]+fl[1]+fl[2] + self.ms_nums*(i+1)
            if i < self.node_nums:
                res[:, l:r] = ob[:, 4, :, i] / CONFIG.estimated_max_lamda
            else:
                res[:, l:r] = ob[:, 5, :, i-self.node_nums] / CONFIG.estimated_max_lamda
        
        for i in range(CONFIG.history_step_length):
                res[:, fl[0]+fl[1]+fl[2]+fl[3]+i] = ob[:, 6, i//self.node_nums, i%self.ms_nums]
        
        return res
    
    def forward(self, ob, a):
        x = self._standardize_state(ob) # 标准化从环境观测到的状态
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.node_nums = CONFIG.node_nums
        self.ms_nums = CONFIG.ms_nums
        self.delta = CONFIG.max_instance_update_num*2+1
        self.feature_length_list = [self.node_nums*self.ms_nums, self.node_nums, self.node_nums, self.ms_nums*CONFIG.history_lamda_length, CONFIG.history_step_length]
        
        self.fc1 = nn.Linear(np.sum(self.feature_length_list), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def _standardize_state(self, ob) -> torch.Tensor:
        """ 标准化状态，支持批次形状，同时展平给DNN输入 """
        batch_size = ob.shape[0]
        total_features = np.sum(self.feature_length_list)
        fl = self.feature_length_list
        res = torch.zeros((batch_size, total_features), dtype=torch.float32, device=CONFIG.device)

        res[:, :fl[0]] = ob[:, 0].view(batch_size, -1) / min(
            CONFIG.node_max_cpu_resource / CONFIG.ms_max_cpu_resource,
            CONFIG.node_min_memory_resource / CONFIG.ms_min_memory_resource
        )
        res[:, fl[0]:fl[0]+fl[1]] = ob[:, 1, 0] / CONFIG.node_max_cpu_resource
        res[:, fl[0]+fl[1]:fl[0]+fl[1]+fl[2]] = ob[:, 2, 0] / CONFIG.node_max_memory_resource
        # res[:, fl[0]+fl[1]+fl[2]:fl[0]+fl[1]+fl[2]+fl[3]] = (ob[:, 3, :, 0] / CONFIG.estimated_max_lamda)
        for i in range(CONFIG.history_lamda_length):
            l = fl[0]+fl[1]+fl[2] + self.ms_nums*i
            r = fl[0]+fl[1]+fl[2] + self.ms_nums*(i+1)
            if i < self.node_nums:
                res[:, l:r] = ob[:, 4, :, i] / CONFIG.estimated_max_lamda
            else:
                res[:, l:r] = ob[:, 5, :, i-self.node_nums] / CONFIG.estimated_max_lamda
        
        for i in range(CONFIG.history_step_length):
                res[:, fl[0]+fl[1]+fl[2]+fl[3]+i] = ob[:, 6, i//self.node_nums, i%self.ms_nums]
        
        return res
    
    def forward(self, ob):
        x = self._standardize_state(ob) # 标准化从环境观测到的状态
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
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

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.AsyncVectorEnv(
        [make_env(i, CONFIG) for i in range(CONFIG.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
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
    rb = ReplayBuffer(int(args.replay_buffer_size))
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=CONFIG.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, actions, rewards, real_next_obs, terminations)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = {"observations": None, "actions": None, "rewards": None, "next_observations": None, "dones": None}
            data["observations"], data["actions"], data["rewards"], data["next_observations"], data["dones"] = rb.sample(args.batch_size)
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

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    envs.close()
    writer.close()

if __name__ == "__main__":
    seed_all(CONFIG.seed)
    train()