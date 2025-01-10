import random
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import os, sys
from datetime import datetime
import collections
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from env import environment, config


CONFIG = config.EnvConfig()

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
    
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCritic(nn.Module):
    def __init__(self, node_nums, ms_nums, max_delta):
        super().__init__()
        self.node_nums = node_nums
        self.ms_nums = ms_nums
        self.feature_length_list = [self.node_nums*self.ms_nums, self.node_nums, self.node_nums, self.ms_nums]
        self.delta = max_delta*2+1

        self.dnn = nn.Sequential(
            layer_init(nn.Linear(np.sum(self.feature_length_list), 1024)),
            nn.ReLU(),
            layer_init(nn.Linear(1024, 1024)),
            nn.ReLU(),
            layer_init(nn.Linear(1024, 1024)),
            nn.ReLU(),
            layer_init(nn.Linear(1024, 1024)),
            nn.ReLU()
        )

        # Actor heads (discrete actions) (node_idx, ms_idx, delta)
        action_size = self.node_nums * self.ms_nums * self.delta
        self.actor = layer_init(nn.Linear(1024, action_size))

    def _standardize_state(self, ob) -> torch.Tensor:
        """ 标准化状态，支持批次形状，同时展平给DNN输入 """
        batch_size = ob.shape[0]
        total_features = np.sum(self.feature_length_list)
        fl = self.feature_length_list
        res = torch.zeros((batch_size, total_features), dtype=torch.float32, device=CONFIG.device)

        res[:, :fl[0]] = ob[:, 0].view(batch_size, -1) / min(
            CONFIG.node_max_cpu_resource / CONFIG.ms_max_cpu_resource,
            CONFIG.node_min_cpu_resource / CONFIG.ms_min_cpu_resource,
            CONFIG.node_max_memory_resource / CONFIG.ms_max_memory_resource,
            CONFIG.node_min_memory_resource / CONFIG.ms_min_memory_resource
        )
        res[:, fl[0]:fl[0]+fl[1]] = ob[:, 1, 0] / CONFIG.node_max_cpu_resource
        res[:, fl[0]+fl[1]:fl[0]+fl[1]+fl[2]] = ob[:, 2, 0] / CONFIG.node_max_memory_resource
        res[:, fl[0]+fl[1]+fl[2]:] = (ob[:, 3, :, 0] / CONFIG.estimated_max_lamda)
        
        data = res.cpu().numpy()    #debug
        return res

    def get_value(self, ob):
        x = self._standardize_state(ob)
        features = self.dnn(x)
        value = self.actor(features)

        return value

    def get_action(self, ob):
        x = self._standardize_state(ob)
        features = self.dnn(x)
        action_prob = F.softmax(self.actor(features), dim=1)

        return action_prob
    
class SACAgent:
    ''' 处理离散动作的SAC算法 '''
    def __init__(self, env: environment.DataCenterEnvironment, config: config.EnvConfig):
        # 策略网络
        self.actor = ActorCritic(env.server_node_nums, env.ms_nums, config.max_instance_update_num).to(CONFIG.device)
        # 第一个Q网络
        self.critic_1 = ActorCritic(env.server_node_nums, env.ms_nums, config.max_instance_update_num).to(CONFIG.device)
        # 第二个Q网络
        self.critic_2 = ActorCritic(env.server_node_nums, env.ms_nums, config.max_instance_update_num).to(CONFIG.device)
        self.target_critic_1 = ActorCritic(env.server_node_nums, env.ms_nums, config.max_instance_update_num).to(CONFIG.device)  # 第一个目标Q网络
        self.target_critic_2 = ActorCritic(env.server_node_nums, env.ms_nums, config.max_instance_update_num).to(CONFIG.device)  # 第二个目标Q网络
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-5)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=1e-5)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=1e-5)
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=1e-5)
        self.target_entropy = -1  # 目标熵的大小
        self.gamma = 0.98
        self.tau = 0.005
        self.device = env.config.device

    def save(self, path, name):
        save_path = os.path.join(path, name)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        
        torch.save(self.actor.state_dict(), save_path)

    def load(self, path):
        load_path = os.path.join(path, "model_SAC.pth")
        self.actor.load_state_dict(torch.load(load_path, weights_only=True))

    def get_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor.get_action(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    # 计算目标Q值,直接用策略网络的输出概率进行期望计算
    def calc_target(self, rewards, next_states, dones):
        next_probs = self.actor.get_action(next_states)
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        q1_value = self.target_critic_1.get_value(next_states)
        q2_value = self.target_critic_2.get_value(next_states)
        min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=True)
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)  # 动作不再是float类型
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_q_values = self.critic_1.get_value(states).gather(1, actions)
        critic_1_loss = torch.mean(
            F.mse_loss(critic_1_q_values, td_target.detach()))
        critic_2_q_values = self.critic_2.get_value(states).gather(1, actions)
        critic_2_loss = torch.mean(
            F.mse_loss(critic_2_q_values, td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新策略网络
        probs = self.actor.get_action(states)
        log_probs = torch.log(probs + 1e-8)
        # 直接根据概率计算熵
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)  #
        q1_value = self.critic_1.get_value(states)
        q2_value = self.critic_2.get_value(states)
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=True)  # 直接根据概率计算期望
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)


def train(agent: SACAgent):
    CONFIG = config.EnvConfig()
    save_path = os.path.join(CONFIG.model_path, datetime.now().strftime("%m%d"), datetime.now().strftime("%H%M%S"), "SAC_las")
    writer = SummaryWriter(save_path)

    env = environment.DataCenterEnvironment(0, CONFIG)
    num_episodes = 200
    replay_buffer = ReplayBuffer(10000)
    minimal_size = 2000
    batch_size = 256

    # TRY NOT TO MODIFY: seeding
    random.seed(CONFIG.seed)
    np.random.seed(CONFIG.seed)
    torch.manual_seed(CONFIG.seed)
    torch.backends.cudnn.deterministic = True
    
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                iteration = i_episode + i * int(num_episodes/10)
                total_reward = []
                total_y = []
                total_Qt = []
                total_delay = {"t_all": [], "t_exe": [], "t_route": []}
                total_vload = []
                total_ns = []
                total_cost = []
                total_node_using_num = []
                total_image_nums = []
                total_rsr = []  # 请求成功率
                total_penalty = []
                state, _ = env.reset()
                done = False
                while not done:
                    action = agent.get_action(state)
                    next_state, reward, done, _,  info = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    total_reward.append(reward)
                    total_y.append(info['y'])
                    total_Qt.append(info['Qt'])
                    total_delay["t_all"].append(info['t_all'])
                    total_delay["t_exe"].append(info['t_exe'])
                    total_delay["t_route"].append(info['t_route'])
                    total_vload.append(info['vload'])
                    total_ns.append(info['ns'])
                    total_cost.append(info['cost'])
                    total_node_using_num.append(info['node_using_num'])
                    total_image_nums.append(info['image_nums'])
                    total_rsr.append(info['request_success_rate'])
                    total_penalty.append(info['penalty'])

                    if done:
                        writer.add_scalar("charts/reward", np.sum(total_reward), iteration)
                        writer.add_scalar("charts/y", np.mean(total_y), iteration)
                        writer.add_scalar("charts/Qt", np.mean(total_Qt), iteration)
                        writer.add_scalar("charts/t_all", np.mean(total_delay["t_all"]), iteration)
                        writer.add_scalar("charts/t_exe", np.mean(total_delay["t_exe"]), iteration)
                        writer.add_scalar("charts/t_route", np.mean(total_delay["t_route"]), iteration)
                        writer.add_scalar("charts/vload", np.mean(total_vload), iteration)
                        writer.add_scalar("charts/ns", np.sum(total_ns), iteration)
                        writer.add_scalar("charts/cost", np.mean(total_cost), iteration)
                        writer.add_scalar("charts/node_using_num", np.mean(total_node_using_num), iteration)
                        writer.add_scalar("charts/image_nums", np.mean(total_image_nums), iteration)
                        writer.add_scalar("charts/rsr", np.mean(total_rsr), iteration)
                        writer.add_scalar("charts/penalty", np.sum(total_penalty), iteration)

                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(np.sum(total_reward))
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-5:])})
                pbar.update(1)

    env.close()
    writer.close()

if __name__ == '__main__':
    _env = environment.DataCenterEnvironment(-1, CONFIG)
    agent = SACAgent(_env, CONFIG)
    train(agent)