import argparse
import os
import random
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smoke-test Gymnasium vector env backends with random actions."
    )
    parser.add_argument("--config", default="twitter_xlargescale")
    parser.add_argument(
        "--backend",
        choices=("sync", "fork", "spawn", "forkserver"),
        default="spawn",
    )
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--shared-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--preinit-cuda",
        action="store_true",
        help="Touch CUDA before creating worker processes, matching PPO/DeepScaler ordering.",
    )
    return parser.parse_args()


def make_env(env_id, config, agent_type):
    from env import environment

    def thunk():
        return environment.DataCenterEnvironment(env_id, config, True, agent_type=agent_type)

    return thunk


def sample_tuple_actions(rng, config, num_envs):
    return np.stack(
        [
            rng.integers(0, config.node_nums, size=num_envs),
            rng.integers(0, config.ms_nums, size=num_envs),
            rng.integers(0, config.max_instance_update_num * 2 + 1, size=num_envs),
        ],
        axis=0,
    )


def main():
    args = parse_args()
    os.environ["AUTOSCALING_CONFIG"] = args.config

    from env import environment

    config = environment.CONFIG
    if args.num_envs is not None:
        config.num_envs = args.num_envs
    if args.seed is not None:
        config.seed = args.seed

    random.seed(config.seed)
    np.random.seed(config.seed)
    rng = np.random.default_rng(config.seed)

    if args.preinit_cuda:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("--preinit-cuda requested, but CUDA is not available")
        torch.manual_seed(config.seed)
        torch.empty((1024, 1024), device=config.device).normal_()
        torch.cuda.synchronize()
        print(f"preinitialized cuda on {config.device}")

    env_fns = [make_env(i, config, "PPO") for i in range(config.num_envs)]
    if args.backend == "sync":
        envs = gym.vector.SyncVectorEnv(env_fns)
    else:
        envs = gym.vector.AsyncVectorEnv(
            env_fns,
            context=args.backend,
            shared_memory=args.shared_memory,
        )

    try:
        obs, _ = envs.reset(seed=config.seed)
        print(
            f"backend={args.backend} shared_memory={args.shared_memory} "
            f"num_envs={config.num_envs} obs_shape={obs.shape}"
        )
        steps = 0
        for iteration in range(args.iterations):
            for _ in range(config.num_steps):
                actions = sample_tuple_actions(rng, config, config.num_envs)
                obs, rewards, terminations, truncations, infos = envs.step(actions)
                steps += config.num_envs
                if np.any(terminations) or np.any(truncations):
                    break
            print(
                f"iteration={iteration + 1} steps={steps} "
                f"reward_mean={float(np.mean(rewards)):.4f} "
                f"y_mean={float(np.mean(infos.get('y', 0))):.4f}"
            )
            envs.reset(seed=config.seed)
    finally:
        envs.close()


if __name__ == "__main__":
    main()
