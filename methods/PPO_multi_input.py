# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass
import copy

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import os, sys 
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from env import environment, config
from methods import Predicter


OBS_KEYS = ["deploy_info", "cpus", "memories", "lamda"]
CONFIG = config.EnvConfig()

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCritic(nn.Module):
    def __init__(self, node_nums, ms_nums, max_delta):
        super().__init__()
        self.node_nums = node_nums
        self.ms_nums = ms_nums
        self.delta = max_delta*2+1

        # CNN for processing deploy_info
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(1, 8, kernel_size=2, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(8, 16, kernel_size=2, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 64, kernel_size=2, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        cnn_output_size = (self.ms_nums-3) * (self.node_nums-3) * 64

        # MLP for processing cpu remaining resources
        self.mlp_cpu = nn.Sequential(
            layer_init(nn.Linear(self.node_nums, 64)),
            nn.ReLU(),
        )

        # MLP for processing memory remaining resources
        self.mlp_mem = nn.Sequential(
            layer_init(nn.Linear(self.node_nums, 64)),
            nn.ReLU(),
        )

        # MLP for processing request lamda
        self.mlp_lamda = nn.Sequential(
            layer_init(nn.Linear(self.ms_nums, 64)),
            nn.ReLU(),
        )

        # Shared feature extractor
        self.shared_feature = nn.Sequential(
            layer_init(nn.Linear(cnn_output_size + 64 * 3, 256)),
            nn.ReLU(),
        )

        # Actor heads (discrete actions)
        self.actor_nodeIdx = layer_init(nn.Linear(256, self.node_nums), std=0.01)
        self.actor_msIdx = layer_init(nn.Linear(256, self.ms_nums), std=0.01)
        self.actor_delta = layer_init(nn.Linear(256, self.delta), std=0.01)

        # Critic for value function
        self.critic = nn.Sequential(
            layer_init(nn.Linear(256, 1)),
        )

    def get_value(self, ob):
        # Process CNN and MLP inputs
        deploy_features = self.cnn(ob[:,0,:,:].unsqueeze(1))
        cpu_features = self.mlp_cpu(ob[:,1,0,:])
        mem_features = self.mlp_mem(ob[:,2,0,:])
        lamda_features = self.mlp_lamda(ob[:,3,:,0])
        combined_features = torch.cat([deploy_features, cpu_features, mem_features, lamda_features], dim=-1)

        # Extract shared features
        shared_features = self.shared_feature(combined_features)
        return self.critic(shared_features)

    def get_action_and_value(self, ob, action=None):
        # Process CNN and MLP inputs
        deploy_features = self.cnn(ob[:,0,:,:].unsqueeze(1))
        cpu_features = self.mlp_cpu(ob[:,1,0,:])
        mem_features = self.mlp_mem(ob[:,2,0,:])
        lamda_features = self.mlp_lamda(ob[:,3,:,0])
        combined_features = torch.cat([deploy_features, cpu_features, mem_features, lamda_features], dim=-1)

        # Extract shared features
        shared_features = self.shared_feature(combined_features)

        # Discrete action logits
        logits_nodeIdx = self.actor_nodeIdx(shared_features)
        logits_msIdx = self.actor_msIdx(shared_features)
        logits_delta = self.actor_delta(shared_features)

        # Probabilities
        probs_nodeIdx = Categorical(logits=logits_nodeIdx)
        probs_msIdx = Categorical(logits=logits_msIdx)
        probs_delta = Categorical(logits=logits_delta)

        if action is None:
            # Sample actions
            action_nodeIdx = probs_nodeIdx.sample()
            action_msIdx = probs_msIdx.sample()
            action_delta = probs_delta.sample()

            # Combine actions
            action = torch.stack([action_nodeIdx, action_msIdx, action_delta], dim=-1)
        else:
            action_nodeIdx = action[:, 0]
            action_msIdx = action[:, 1]
            action_delta = action[:, 2]

        # Log probabilities，同等权重的直接相加
        logprob_nodeIdx = probs_nodeIdx.log_prob(action_nodeIdx)
        logprob_msIdx = probs_msIdx.log_prob(action_msIdx)
        logprob_delta = probs_delta.log_prob(action_delta)
        logprob = logprob_nodeIdx + logprob_msIdx + logprob_delta

        # Entropies
        entropy = probs_nodeIdx.entropy() + probs_msIdx.entropy() + probs_delta.entropy()

        return action, logprob, entropy, self.critic(shared_features)


class PPOAgent(object):
    def __init__(self, env: environment.DataCenterEnvironment, config: config.EnvConfig):
        # Basic config
        self.env = env
        self.config = config
        self.actorcrtic = ActorCritic(env.server_node_nums, env.ms_nums, config.max_instance_update_num).to(CONFIG.device)
        self.optimizer = optim.Adam(self.actorcrtic.parameters(), lr=config.lr, eps=1e-5)
        # params
        self.gamma = 0.99
        self.gae_lambda = 0.95

def make_env(env_id, config):
    def thunk():
        env = environment.DataCenterEnvironment(env_id, config)
        return env

    return thunk

def train(agent: PPOAgent):
    CONFIG = config.EnvConfig()
    writer = SummaryWriter(f"runs/PPO_multi_input")

    # TRY NOT TO MODIFY: seeding
    random.seed(CONFIG.seed)
    np.random.seed(CONFIG.seed)
    torch.manual_seed(CONFIG.seed)
    torch.backends.cudnn.deterministic = True

    predicter = Predicter.SMAPredictor(CONFIG.ms_nums, CONFIG.predicter_window_size)
    envs = gym.vector.SyncVectorEnv(
        [make_env(i, CONFIG) for i in range(CONFIG.num_envs)],
    )

    # Storage setup
    obs = torch.zeros((CONFIG.num_steps, CONFIG.num_envs) + envs.single_observation_space.shape).to(CONFIG.device)
    actions = torch.zeros((CONFIG.num_steps, CONFIG.num_envs, len(envs.single_action_space))).to(CONFIG.device)
    logprobs = torch.zeros((CONFIG.num_steps, CONFIG.num_envs)).to(CONFIG.device)
    rewards = torch.zeros((CONFIG.num_steps, CONFIG.num_envs)).to(CONFIG.device)
    dones = torch.zeros((CONFIG.num_steps, CONFIG.num_envs)).to(CONFIG.device)
    values = torch.zeros((CONFIG.num_steps, CONFIG.num_envs)).to(CONFIG.device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()

    for iteration in range(1, CONFIG.num_iterations + 1):
        total_reward = []
        total_congested_ms_nums = []
        total_ns = []
        total_image_nums = []
        next_obs, _ = envs.reset(seed=CONFIG.seed)
        next_obs = torch.Tensor(next_obs).to(CONFIG.device)
        next_done = torch.zeros(CONFIG.num_envs).to(CONFIG.device)
        for step in range(0, CONFIG.num_steps):
            global_step += CONFIG.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.actorcrtic.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy().T)        # 传入step后是按列截取的，而我们期望是按行截取的，故转置一下
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(CONFIG.device).view(-1)
            next_obs = torch.Tensor(next_obs).to(CONFIG.device)
            next_done = torch.Tensor(next_done).to(CONFIG.device)

            total_reward.append(np.sum(reward))
            total_congested_ms_nums.append(np.mean(infos['congested_ms_nums']))
            total_ns.append(np.mean(infos['ns']))
            total_image_nums.append(np.mean(infos['image_nums']))

            if terminations[0]:
                print(f"Iteration: {iteration}, Total Reward: {np.sum(total_reward)}")
                writer.add_scalar("charts/total_reward", np.sum(total_reward), iteration)
                writer.add_scalar("charts/total_congested_ms_nums", np.sum(total_congested_ms_nums), iteration)
                writer.add_scalar("charts/total_ns", np.sum(total_ns), iteration)
                writer.add_scalar("charts/total_image_nums", np.sum(total_image_nums), iteration)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.actorcrtic.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(CONFIG.device)
            lastgaelam = 0
            for t in reversed(range(CONFIG.num_steps)):
                if t == CONFIG.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + CONFIG.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + CONFIG.gamma * CONFIG.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + (len(envs.single_action_space),))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(CONFIG.batch_size)
        clipfracs = []
        for epoch in range(CONFIG.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, CONFIG.batch_size, CONFIG.minibatch_size):
                end = start + CONFIG.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.actorcrtic.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > CONFIG.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if CONFIG.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CONFIG.clip_coef, 1 + CONFIG.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if CONFIG.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -CONFIG.clip_coef,
                        CONFIG.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - CONFIG.ent_coef * entropy_loss + v_loss * CONFIG.vf_coef

                agent.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.actorcrtic.parameters(), CONFIG.max_grad_norm)
                agent.optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", agent.optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
    print("success")

if __name__ == "__main__":
    _env = environment.DataCenterEnvironment(-1, CONFIG)    # 只用于定义agent，不参与实际训练
    agent = PPOAgent(_env, CONFIG)
    train(agent)