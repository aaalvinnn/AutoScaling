"""
DeepScalerGNN: GNN-based Actor-Critic for PPO training.
 
Architecture overview:
  Input: (batch, 7, ms_nums, node_nums) observation
    → GraphLearn produces adaptive adjacency from MS2MS_data_graph
    → DeepScalerGNNEncoder (2 AdapGL blocks, configurable hidden_dim)
    → DNN (MLP, configurable dnn_hidden × dnn_layers)
    → Actor heads (node, ms, delta) + Critic head

Default parameters (hidden_dim=64, dnn_hidden=512, dnn_layers=3) give
~1.16M params on twitter_largescale, close to LGDRL's ~0.66M parameter budget.
 
No external GNN libraries required (pure PyTorch).
"""

import os
import random
import time
import argparse
from datetime import datetime
import shutil
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import inspect
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

for _i in range(1, len(sys.argv)):
    if sys.argv[_i] == "--config" and _i + 1 < len(sys.argv):
        os.environ["AUTOSCALING_CONFIG"] = sys.argv[_i + 1]
        sys.argv.pop(_i); sys.argv.pop(_i)
        break

from env import environment
from methods.vector_env_backend import make_vector_env as make_training_vector_env
from methods.train_checkpoint import (
    save_train_state, load_train_state,
    snapshot_rng, restore_rng,
    reward_scaler_to_dict, load_reward_scaler,
    find_latest_resume_dir,
)

CONFIG = environment.CONFIG


# ── weight initialization (matches PPO_dnn.py) ──────────────────────────────

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ═══════════════════════════════════════════════════════════════════════════════
# 1. GraphConv – GCN-type graph convolution layer
# ═══════════════════════════════════════════════════════════════════════════════

class GraphConv(nn.Module):
    """Graph convolution layer (GCN type).

    Computes H' = ReLU(A @ H @ W  +  H @ W_self)
    where A is the (num_nodes, num_nodes) adjacency matrix.

    Args:
        f_in:   Input feature dimension.
        f_out:  Output feature dimension.
    """

    def __init__(self, f_in: int, f_out: int):
        super().__init__()
        self.w_conv = nn.Linear(f_in, f_out, bias=False)
        self.w_self = nn.Linear(f_in, f_out)

    def forward(self, x: torch.Tensor, adj_mx: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x:      Node features  (batch, num_nodes, f_in).
            adj_mx: Adjacency matrix  (num_nodes, num_nodes).
        Returns:
            (batch, num_nodes, f_out).
        """
        h = torch.matmul(adj_mx, x)         # A @ H          (batch, N, f_in)
        h = self.w_conv(h)                  # A @ H @ W_conv  (batch, N, f_out)
        h = h + self.w_self(x)              # + H @ W_self
        return torch.relu(h)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. GraphLearn – adaptive graph learning module
# ═══════════════════════════════════════════════════════════════════════════════

class GraphLearn(nn.Module):
    """Adaptive graph learning module.

    Learns an adjacency matrix by combining a prior graph (e.g. MS2MS_data_graph)
    with a learned asymmetric component, fused via per-element attention.

    Args:
        num_nodes:        Number of graph nodes (= ms_nums).
        init_feature_num: Dimension of learnable node embeddings.
    """

    def __init__(self, num_nodes: int, init_feature_num: int):
        super().__init__()
        self.beta = nn.Parameter(torch.rand(num_nodes))
        self.w1 = nn.Parameter(torch.zeros((num_nodes, init_feature_num)))
        self.w2 = nn.Parameter(torch.zeros((num_nodes, init_feature_num)))
        self.attn = nn.Conv2d(2, 1, kernel_size=1)

        # register the initial adjacency; can be replaced via set_adj()
        self.register_buffer("init_adj", torch.eye(num_nodes))

    def set_adj(self, adj_mx: torch.Tensor):
        """Set the initial (prior) adjacency matrix."""
        self.init_adj = adj_mx.to(self.init_adj.device, dtype=self.init_adj.dtype)

    def forward(self) -> torch.Tensor:
        """Compute adaptive adjacency matrix.

        Returns:
            (num_nodes, num_nodes) symmetric-normalized adaptive adjacency.
        """
        adj_mx = self.init_adj
        # asymmetric learned component
        new_adj = torch.mm(self.w1, self.w2.T) - torch.mm(self.w2, self.w1.T)
        new_adj = torch.relu(new_adj + torch.diag(self.beta))

        # per-element attention fusion with prior
        stacked = torch.stack((new_adj, adj_mx), dim=0).unsqueeze(0)  # (1, 2, N, N)
        attn = torch.sigmoid(self.attn(stacked).squeeze())             # (N, N)
        fused = attn * new_adj + (1 - attn) * adj_mx

        # symmetric normalisation  D^{-0.5} A D^{-0.5}
        d = fused.sum(dim=1) ** (-0.5)
        d = torch.nan_to_num(d, nan=0.0, posinf=0.0)
        fused = d.view(-1, 1) * fused * d
        return fused


# ═══════════════════════════════════════════════════════════════════════════════
# 3. AdapGLBlock – single AdapGL block
# ═══════════════════════════════════════════════════════════════════════════════

class AdapGLBlock(nn.Module):
    """A single AdapGL block combining temporal attention, graph convolution,
    temporal convolution, and residual connection.

    Args:
        num_nodes:  Number of graph nodes (= ms_nums).
        channels:   Number of feature channels (same input & output).
        graph_conv: Shared GraphConv module.
    """

    def __init__(self, num_nodes: int, channels: int, graph_conv: GraphConv):
        super().__init__()
        # temporal self-attention over the "time" (channel) axis
        self.temporal_attn = nn.MultiheadAttention(
            channels, num_heads=4, batch_first=True, dropout=0.0
        )
        self.norm1 = nn.LayerNorm(channels)

        # shared graph convolution (operates on node dimension)
        self.graph_conv = graph_conv

        # temporal 1-D convolution along the time axis
        self.time_conv = nn.Conv2d(
            channels, channels, kernel_size=(1, 3), padding=(0, 1)
        )
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor,
                adj_mx: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x:      (batch, num_nodes, time_steps, channels).
            adj_mx: (num_nodes, num_nodes) adjacency matrix.
        Returns:
            (batch, num_nodes, time_steps, channels).
        """
        B, N, T, C = x.shape
        residual = x

        x_t = x.reshape(B * N, T, C)
        attn_out, _ = self.temporal_attn(x_t, x_t, x_t)
        x_t = self.norm1(x_t + attn_out)
        x_t = x_t.reshape(B, N, T, C)

        x_flat = x_t.reshape(B * T, N, C)
        x_g = self.graph_conv(x_flat, adj_mx)
        x_g = x_g.reshape(B, T, N, C).permute(0, 2, 1, 3)

        x_c = x_g.permute(0, 3, 1, 2)
        x_c = self.time_conv(x_c)
        x_c = torch.relu(x_c)
        x_c = x_c.permute(0, 2, 3, 1)

        out = x_c + residual
        out = self.norm2(out.reshape(-1, C))
        return out.reshape(B, N, T, C)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DeepScalerGNNEncoder – stack of AdapGL blocks
# ═══════════════════════════════════════════════════════════════════════════════

class DeepScalerGNNEncoder(nn.Module):
    """Feature extractor: input projection + 2 AdapGL blocks.

    Input:  (batch, ms_nums, time_steps=7, features=node_nums)
    Output: (batch, ms_nums * hidden_dim)   flattened node features.

    Args:
        ms_nums:      Number of graph nodes (microservices).
        node_nums:    Input feature dimension per node.
        hidden_dim:   Hidden channel dimension (default 64).
        graph_conv:   Shared GraphConv module.
    """

    def __init__(self, ms_nums: int, node_nums: int,
                 hidden_dim: int = 64, graph_conv: GraphConv = None):
        super().__init__()
        self.ms_nums = ms_nums
        self.hidden_dim = hidden_dim

        # 1×1 input projection  (features → hidden_dim) per time-step
        self.input_proj = nn.Conv2d(node_nums, hidden_dim, kernel_size=1)

        if graph_conv is not None:
            self.graph_conv = graph_conv
        else:
            self.graph_conv = GraphConv(hidden_dim, hidden_dim)

        # stack 2 AdapGL blocks
        self.block1 = AdapGLBlock(ms_nums, hidden_dim, self.graph_conv)
        self.block2 = AdapGLBlock(ms_nums, hidden_dim, self.graph_conv)

    def forward(self, x: torch.Tensor,
                adj_mx: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x:      (batch, ms_nums, time_steps=7, features=node_nums).
            adj_mx: (ms_nums, ms_nums) adaptive adjacency.
        Returns:
            (batch, ms_nums * hidden_dim).
        """
        B = x.shape[0]

        # permute to (B, features, nodes, time) for Conv2d input projection
        x = x.permute(0, 3, 1, 2)                      # (B, node_nums, N, T)
        x = self.input_proj(x)                          # (B, hidden_dim, N, T)
        x = x.permute(0, 2, 3, 1)                       # (B, N, T, hidden_dim)

        # 2 AdapGL blocks
        x = self.block1(x, adj_mx)
        x = self.block2(x, adj_mx)

        # global mean pooling over time → (B, N, hidden_dim)
        x = x.mean(dim=2)

        # flatten → (B, N * hidden_dim)
        return x.reshape(B, -1)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ActorCriticGNN – Actor-Critic with GNN encoder
# ═══════════════════════════════════════════════════════════════════════════════

class ActorCriticGNN(nn.Module):
    """Actor-Critic model with GNN feature encoder + DNN reasoning layers.

    Architecture: Observation → GNN Encoder → DNN (MLP) → Actor/Critic heads.
    The DNN bridges the GNN output to the policy/value heads, matching the
    parameter budget of LGDRL's 7-layer MLP (≈1.7M params on largescale).

    Defaults (hidden_dim=64, dnn_hidden=512, dnn_layers=3) give ≈1.16M params
    on twitter_largescale, close to LGDRL's ≈0.66M parameter budget.

    Args:
        config:      EnvConfig dataclass with ms_nums, node_nums, device, etc.
        hidden_dim:  GNN encoder channel dimension (default 64).
        dnn_hidden:  DNN hidden layer width (default 512, matches LGDRL).
        dnn_layers:  Total DNN Linear layers including input projection (default 3).
    """

    def __init__(self, config, hidden_dim: int = 64,
                 dnn_hidden: int = 512, dnn_layers: int = 3):
        super().__init__()
        self.config = config
        self.node_nums = config.node_nums
        self.ms_nums = config.ms_nums
        self.delta_size = config.max_instance_update_num * 2 + 1

        # ── adaptive graph learner ──
        self.graph_learn = GraphLearn(config.ms_nums, config.node_nums)

        # ── GNN encoder ──
        shared_gc = GraphConv(hidden_dim, hidden_dim)
        self.encoder = DeepScalerGNNEncoder(
            config.ms_nums, config.node_nums, hidden_dim, shared_gc
        )

        encoder_out = config.ms_nums * hidden_dim

        # ── DNN: GNN features → policy/value representation ──
        # Matches LGDRL's MLP depth for parameter parity
        dnn_modules = []
        dnn_modules.append(layer_init(nn.Linear(encoder_out, dnn_hidden)))
        dnn_modules.append(nn.Tanh())
        for _ in range(dnn_layers - 1):
            dnn_modules.append(layer_init(nn.Linear(dnn_hidden, dnn_hidden)))
            dnn_modules.append(nn.Tanh())
        self.dnn = nn.Sequential(*dnn_modules)

        # ── actor heads (3 discrete heads) ──
        self.actors = nn.ModuleList([
            layer_init(nn.Linear(dnn_hidden, self.node_nums), std=0.01),
            layer_init(nn.Linear(dnn_hidden, self.ms_nums), std=0.01),
            layer_init(nn.Linear(dnn_hidden, self.delta_size), std=0.01),
        ])

        # ── critic head ──
        self.critic = nn.Sequential(
            layer_init(nn.Linear(dnn_hidden, 1)),
        )

    # ── helpers ────────────────────────────────────────────────────────────

    def get_adj(self) -> torch.Tensor:
        """Return the current adaptive adjacency matrix."""
        return self.graph_learn()

    def _standardize_state(self, ob: torch.Tensor) -> torch.Tensor:
        """Normalise the raw observation before feeding to the GNN.

        Input:  (batch, 7, ms_nums, node_nums)  raw observation.
        Output: (batch, ms_nums, 7, node_nums)  normalised, permuted.
        """
        cfg = self.config
        batch_size = ob.shape[0]
        out = torch.zeros_like(ob)

        # channel 0: deployment info (already count-based, keep as-is)
        out[:, 0] = ob[:, 0] / max(
            cfg.node_max_cpu_resource / max(cfg.ms_max_cpu_resource, 1),
            cfg.node_min_memory_resource / max(cfg.ms_min_memory_resource, 1),
            1,
        )

        # channel 1: CPU resources
        out[:, 1] = ob[:, 1] / cfg.node_max_cpu_resource

        # channel 2: memory resources
        out[:, 2] = ob[:, 2] / cfg.node_max_memory_resource

        # channel 3: predicted arrival rates
        out[:, 3] = ob[:, 3] / max(cfg.estimated_max_lamda, 1)

        # channel 4-5: historical arrival rates
        out[:, 4] = ob[:, 4] / max(cfg.estimated_max_lamda, 1)
        out[:, 5] = ob[:, 5] / max(cfg.estimated_max_lamda, 1)

        # channel 6: timestep one-hot (already 0/1)
        out[:, 6] = ob[:, 6]

        # permute: (batch, 7, ms_nums, node_nums) → (batch, ms_nums, 7, node_nums)
        return out.permute(0, 2, 1, 3)

    # ── PPO-compatible API ─────────────────────────────────────────────────

    def get_value(self, ob: torch.Tensor) -> torch.Tensor:
        ob = self._standardize_state(ob)
        adj = self.get_adj()
        features = self.encoder(ob, adj)
        features = self.dnn(features)
        return self.critic(features)

    def get_action_and_value(self, ob: torch.Tensor,
                              action: torch.Tensor = None,
                              pre_standardized: bool = False
                              ) -> tuple:
        if not pre_standardized:
            ob = self._standardize_state(ob)

        adj = self.get_adj()
        features = self.encoder(ob, adj)
        features = self.dnn(features)

        logits = [actor(features) for actor in self.actors]
        probs = [Categorical(logits=logit) for logit in logits]

        if action is None:
            action = torch.stack([prob.sample() for prob in probs], dim=-1)

        logprob = sum(prob.log_prob(action[:, i])
                      for i, prob in enumerate(probs))
        entropy = sum(prob.entropy() for prob in probs)

        return action, logprob, entropy, self.critic(features)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. DeepScalerAgent – Agent wrapper with save / load
# ═══════════════════════════════════════════════════════════════════════════════

class DeepScalerAgent:
    """Agent that wraps ActorCriticGNN for PPO training.

    Follows the same interface as PPOAgent from PPO_dnn.py.

    Args:
        config:        EnvConfig dataclass.
        ms2ms_graph:   (ms_nums, ms_nums) numpy adjacency matrix
                       (the MS2MS_data_graph from the environment).
        hidden_dim:    GNN encoder channel dimension (default 64).
        dnn_hidden:    DNN hidden layer width (default 512).
        dnn_layers:    Total DNN Linear layers (default 3).
    """

    def __init__(self, config, ms2ms_graph: np.ndarray = None,
                 hidden_dim: int = 64, dnn_hidden: int = 512, dnn_layers: int = 3):
        self.config = config
        self.actorcrtic = ActorCriticGNN(
            config, hidden_dim=hidden_dim, dnn_hidden=dnn_hidden, dnn_layers=dnn_layers
        ).to(config.device)
        self.optimizer = optim.Adam(
            self.actorcrtic.parameters(), lr=config.lr, eps=1e-5
        )

        # set initial adjacency matrix in GraphLearn
        if ms2ms_graph is not None:
            adj_t = torch.from_numpy(ms2ms_graph).float().to(config.device)
            # normalise the initial adjacency
            d = adj_t.sum(dim=1) ** (-0.5)
            d = torch.nan_to_num(d, nan=0.0, posinf=0.0)
            adj_t = d.view(-1, 1) * adj_t * d
            self.actorcrtic.graph_learn.set_adj(adj_t)

    def save(self, path: str, name: str):
        """Save model state dict.

        Args:
            path:  Directory path.
            name:  Filename (e.g. "model_best.pth").
        """
        save_path = os.path.join(path, name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.actorcrtic.state_dict(), save_path)

    def load(self, path: str):
        """Load model state dict from checkpoint.

        Args:
            path: Directory containing "model.pth", or direct path to a .pth file.
        """
        if path.endswith('.pth'):
            load_path = path
        else:
            load_path = os.path.join(path, "model.pth")
        self.actorcrtic.load_state_dict(
            torch.load(load_path, map_location=self.config.device, weights_only=True)
        )

    def get_action(self, ob: np.ndarray) -> np.ndarray:
        """Inference: single observation → action.

        Args:
            ob: (7, ms_nums, node_nums) numpy array.
        Returns:
            (3,) numpy array  [node_id, ms_id, delta].
        """
        ob_t = torch.Tensor(ob).unsqueeze(0).to(self.config.device)
        with torch.no_grad():
            action, _, _, _ = self.actorcrtic.get_action_and_value(ob_t)
        return action.cpu().numpy()[0]


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Training function
# ═══════════════════════════════════════════════════════════════════════════════

def make_env(env_id, config, agent_type="DeepScaler"):
    def thunk():
        return environment.DataCenterEnvironment(env_id, config, True, agent_type=agent_type)
    return thunk


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class RewardScaler:
    def __init__(self, record_epoch, epsilon=1e-8):
        self.buffer = []
        self.mean = None
        self.var = None
        self.count = 0
        self.record_epoch = record_epoch
        self.epsilon = epsilon

    def record(self, reward):
        """ 记录奖励 """
        self.buffer.append(reward)

    def compute_mean_and_var(self):
        if len(self.buffer) == 0:
            self.mean = 0.0
            self.var = 1.0
        else:
            self.mean = np.mean(self.buffer)
            self.var = np.var(self.buffer)

    def reward_shaping(self, reward):
        """ 通过前若干轮的探索轮次计算均值和方差，然后归一化 """
        if self.mean is None or self.var is None:
            self.compute_mean_and_var()
        new_reward = (reward - self.mean) / (self.var ** 0.5 + self.epsilon)
        return new_reward


def save_config(save_path, config):
    config_dir = os.path.join(save_path, "config")
    os.makedirs(config_dir, exist_ok=True)
    shutil.copy(Path(__file__), config_dir)
    config_path = inspect.getfile(config.__class__)
    shutil.copy(config_path, config_dir)


def train(config, agent: DeepScalerAgent, agent_type: str = "DeepScaler", save_every: int = 250, resume_dir=None):
    # ── 续跑：写进同一目录（TensorBoard 曲线连续），否则用新时间戳目录 ──
    resume_state = None
    if resume_dir is not None:
        ckpt_file = os.path.join(resume_dir, "train_state.pt")
        if os.path.exists(ckpt_file):
            print(f"[resume] loading train_state.pt from {resume_dir}")
            resume_state = load_train_state(ckpt_file, map_location=config.device)
            save_path = resume_dir
        else:
            print(f"[resume] {ckpt_file} 不存在，回退冷启动")
            save_path = os.path.join(
                config.model_path, config.config_name,
                datetime.now().strftime("%m%d"), datetime.now().strftime("%H%M"), "DeepScaler"
            )
    else:
        save_path = os.path.join(
            config.model_path, config.config_name,
            datetime.now().strftime("%m%d"), datetime.now().strftime("%H%M"), "DeepScaler"
        )

    # 已训练完成（iteration 达到 num_iterations）则直接退出，不 spawn env、不空跑
    if resume_state is not None and int(resume_state.get("iteration", 0)) >= config.num_iterations:
        print(f"[done] iteration={int(resume_state['iteration'])} 已达 num_iterations={config.num_iterations}，训练完成，停止", flush=True)
        return

    writer = SummaryWriter(save_path)
    if resume_state is None:
        save_config(save_path, config)

    # spawn: async 步进（快），且避开 fork-after-CUDA/numpy 的 worker 腐蚀。
    # 可用 AUTOSCALING_VECTOR_BACKEND=fork|sync|forkserver 覆盖做对照。
    envs = make_training_vector_env(
        [make_env(i, config, agent_type=agent_type) for i in range(config.num_envs)],
        default_backend="spawn",
    )

    # reward scaling
    reward_shaping_record_epoch = getattr(config, 'reward_shaping_record_epoch', 100)
    reward_scaler = RewardScaler(record_epoch=reward_shaping_record_epoch)

    obs = torch.zeros((config.num_steps, config.num_envs) + envs.single_observation_space.shape).to(config.device)
    actions = torch.zeros((config.num_steps, config.num_envs, len(envs.single_action_space))).to(config.device)
    logprobs = torch.zeros((config.num_steps, config.num_envs)).to(config.device)
    rewards = torch.zeros((config.num_steps, config.num_envs)).to(config.device)
    dones = torch.zeros((config.num_steps, config.num_envs)).to(config.device)
    values = torch.zeros((config.num_steps, config.num_envs)).to(config.device)

    global_step = 0
    start_time = time.time()
    best_y = np.inf
    start_iteration = 1
    record_reward = []

    # ── 恢复全状态：权重 + 优化器 + reward shaping + 计数器 + RNG ──
    if resume_state is not None:
        agent.actorcrtic.load_state_dict(resume_state["model"])
        agent.optimizer.load_state_dict(resume_state["optimizer"])
        load_reward_scaler(reward_scaler, resume_state["reward_scaler"])
        start_iteration = int(resume_state["iteration"]) + 1
        global_step = int(resume_state.get("global_step", 0))
        best_y = float(resume_state.get("best_y", np.inf))
        restore_rng(resume_state.get("rng"))
        print(f"[resume] 从 iteration={start_iteration} 续跑 | "
              f"reward_shaping mean={reward_scaler.mean} var={reward_scaler.var} | "
              f"optimizer state={len(agent.optimizer.state_dict().get('state', {}))} 组")

    with tqdm(total=config.num_iterations, desc="Training DeepScaler", unit="it", initial=start_iteration - 1) as pbar:
        for iteration in range(start_iteration, config.num_iterations + 1):
            total_reward = []
            total_y = []
            total_cost = []
            total_delay = []
            total_rsr = []

            next_obs, _ = envs.reset(seed=config.seed)
            next_obs = torch.Tensor(next_obs).to(config.device)
            next_done = torch.zeros(config.num_envs).to(config.device)

            for step in range(0, config.num_steps):
                global_step += config.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = agent.actorcrtic.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                try:
                    action_np = action.detach().cpu().numpy().T
                    next_obs, reward, terminations, truncations, infos = envs.step(action_np)
                except Exception:
                    print("shape error, pass!")
                    break

                next_done = np.logical_or(terminations, truncations)
                next_obs = torch.Tensor(next_obs).to(config.device)
                next_done = torch.Tensor(next_done).to(config.device)

                if iteration < reward_shaping_record_epoch:
                    reward_scaler.record(reward)
                    continue

                reward = reward_scaler.reward_shaping(reward)
                rewards[step] = torch.tensor(reward).to(config.device).view(-1)

                total_reward.append(np.mean(reward))
                total_y.append(np.mean(infos['y']))
                total_cost.append(np.mean(infos['cost']))
                total_delay.append(np.mean(infos['t_all']))
                total_rsr.append(np.mean(infos['request_success_rate']))

            if total_y:
                writer.add_scalar("charts/reward", np.sum(total_reward), iteration)
                writer.add_scalar("charts/y", np.mean(total_y), iteration)
                writer.add_scalar("charts/cost", np.mean(total_cost), iteration)
                writer.add_scalar("charts/t_all", np.mean(total_delay), iteration)
                writer.add_scalar("charts/rsr", np.mean(total_rsr), iteration)

            if iteration < reward_shaping_record_epoch:
                continue

            if best_y > np.mean(total_y):
                best_y = np.mean(total_y)
                agent.save(save_path, "model_best.pth")
                agent.save(save_path, "model.pth")
            if iteration % save_every == 0:
                agent.save(save_path, f"model_{iteration}.pth")
            # 全状态 checkpoint（原子写）：权重+优化器+reward shaping+计数器+RNG，供崩溃续跑
            save_train_state(
                os.path.join(save_path, "train_state.pt"),
                {
                    "model": agent.actorcrtic.state_dict(),
                    "optimizer": agent.optimizer.state_dict(),
                    "reward_scaler": reward_scaler_to_dict(reward_scaler),
                    "iteration": iteration,
                    "global_step": global_step,
                    "best_y": float(best_y),
                    "rng": snapshot_rng(),
                },
            )

            with torch.no_grad():
                next_value = agent.actorcrtic.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(config.device)
                lastgaelam = 0
                for t in reversed(range(config.num_steps)):
                    if t == config.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + config.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + (len(envs.single_action_space),))
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            b_obs_std = agent.actorcrtic._standardize_state(b_obs)
            b_actions_long = b_actions.long()

            b_inds = np.arange(config.batch_size)
            clipfracs = []
            for epoch in range(config.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, config.batch_size, config.minibatch_size):
                    end = start + config.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.actorcrtic.get_action_and_value(
                        b_obs_std[mb_inds], b_actions_long[mb_inds], pre_standardized=True
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > config.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if config.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    newvalue = newvalue.view(-1)
                    if config.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds], -config.clip_coef, config.clip_coef
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - config.ent_coef * entropy_loss + v_loss * config.vf_coef

                    agent.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.actorcrtic.parameters(), config.max_grad_norm)
                    agent.optimizer.step()

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            pbar.update(1)
            pbar.set_postfix({
                "reward": f"{np.mean(total_reward):.2f}",
                "y": f"{np.mean(total_y):.2f}",
                "cost": f"{np.mean(total_cost):.2f}"
            })

    envs.close()
    writer.close()
    print(f"success: env={config.config_name} DeepScaler training complete")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="DeepScaler GNN Training")
    parser.add_argument("--q-max", type=int, help="Q_max parameter override")
    parser.add_argument("--hidden-dim", type=int, default=64, help="GNN encoder hidden channels")
    parser.add_argument("--dnn-hidden", type=int, default=512, help="DNN hidden layer width (default 512, matches LGDRL)")
    parser.add_argument("--dnn-layers", type=int, default=3, help="Total DNN Linear layers (default 3)")
    parser.add_argument("--device", help="CUDA device override")
    parser.add_argument("--seed", type=int, help="Random seed override")
    parser.add_argument("--resume", help="Resume from checkpoint")
    parser.add_argument("--auto-resume", action="store_true", help="自动从最近一次 train_state.pt 续跑（供外层重启循环调用）")
    parser.add_argument("--num-iterations", type=int, help="Override num_iterations")
    parser.add_argument("--lyapunov", action="store_true", help="Use Lyapunov reward (-y) instead of weighted-sum")
    parser.add_argument("--save-every", type=int, default=250, help="Save checkpoint every N iterations (default 250)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args() 
    config = CONFIG

    if args.q_max is not None:
        config.Q_max = args.q_max
    if args.device:
        config.device = args.device
    if args.seed is not None:
        config.seed = args.seed
    if args.num_iterations is not None:
        config.num_iterations = args.num_iterations

    seed_all(config.seed)

    temp_env = environment.DataCenterEnvironment(0, config, is_train=True)
    temp_env.reset(seed=config.seed)
    ms2ms_graph = temp_env.MS2MS_data_graph

    agent = DeepScalerAgent(config, ms2ms_graph,
                            hidden_dim=args.hidden_dim,
                            dnn_hidden=args.dnn_hidden,
                            dnn_layers=args.dnn_layers)

    if args.resume:
        agent.load(args.resume)

    # --auto-resume: 从最近一次 train_state.pt 续跑（全状态，覆盖上面的 --resume 权重加载）
    resume_dir = find_latest_resume_dir(config, "DeepScaler") if args.auto_resume else None

    # choose agent_type string for reward function
    env_agent_type = "DeepScaler-Lyapunov" if args.lyapunov else "DeepScaler"

    train(config, agent, agent_type=env_agent_type, save_every=args.save_every, resume_dir=resume_dir)
