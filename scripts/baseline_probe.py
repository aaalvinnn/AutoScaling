"""快速探针：只跑启发式 baseline（HPA / ProScaler / NoScaling），输出 rsr/cost/delay。

用法：
    conda run -n as python scripts/baseline_probe.py [config_name]
默认 config = twitter_xlargescale
"""
import sys
import os

# 仓库根加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 必须在 import env 之前选定 config
config_name = sys.argv[1] if len(sys.argv) > 1 else "twitter_xlargescale"
os.environ["AUTOSCALING_CONFIG"] = config_name

import numpy as np
from env.datastruct import CONFIG
from env.environment import DataCenterEnvironment
from methods import HPA, ProScaling, NoScaling

cfg = CONFIG
print(f"{'='*60}")
print(f"Baseline Probe: {cfg.config_name}")
print(f"  nodes={cfg.node_nums}  ms={cfg.ms_nums}  flows={cfg.request_flow_nums}  C={cfg.C}")
print(f"  time_slots={cfg.time_slot_end}  seed={cfg.seed}")
print(f"{'='*60}")

# 创建环境和 agent
agent_names = ["NoScaling", "HPA", "ProScaler"]
envs = [DataCenterEnvironment(i, cfg) for i in range(len(agent_names))]
agents = [
    NoScaling.NoScalingAgent(envs[0]),
    HPA.HPA(envs[1]),
    ProScaling.ProScalingAgent(envs[2]),
]

# 关键指标收集
metrics = {name: {"rsr": [], "cost": [], "t_all": [], "y": [], "t_exe": [], "t_route": [],
                 "image_nums": [], "node_using_num": []}
           for name in agent_names}

total_steps = cfg.time_slot_end  # 288 slots (24h × 5min)
states = [env.reset(seed=cfg.seed)[0] for env in envs]
dones = [False] * len(agents)

for step in range(total_steps):
    for i, (agent, env) in enumerate(zip(agents, envs)):
        if not dones[i]:
            action = agent.get_action(states[i])
            next_state, reward, done, _, info = env.step(action)
            states[i] = next_state
            dones[i] = done

            name = agent_names[i]
            m = metrics[name]
            m["rsr"].append(info.get("request_success_rate", 0))
            m["cost"].append(info.get("cost", 0))
            m["t_all"].append(info.get("t_all", 0))
            m["y"].append(info.get("y", 0))
            m["t_exe"].append(info.get("t_exe", 0))
            m["t_route"].append(info.get("t_route", 0))
            m["image_nums"].append(info.get("image_nums", 0))
            m["node_using_num"].append(info.get("node_using_num", 0))

# 输出结果
n_agents = len(agent_names)
header = f'{"指标":12s}' + ''.join([f' | {name:>10s}' for name in agent_names])
print(f'\n{"="*(15+13*n_agents)}')
print(header)
print('-'*(15+13*n_agents))

key_metrics = [
    ("rsr (请求成功率)", "rsr"),
    ("cost (总开销)", "cost"),
    ("delay (总延迟)", "t_all"),
    ("y (Lyapunov目标)", "y"),
    ("exe_delay (执行延迟)", "t_exe"),
    ("route_delay (路由延迟)", "t_route"),
    ("实例总数", "image_nums"),
    ("使用节点数", "node_using_num"),
]

for label, key in key_metrics:
    vals = []
    for name in agent_names:
        arr = np.array(metrics[name][key])
        mean_val = arr.mean()
        std_val = arr.std()
        vals.append(f"{mean_val:.3f}")
    row = f'{label:12s}' + ''.join([f' | {v:>10s}' for v in vals])
    print(row)

print(f'\n--- 统计细节 (mean ± std over {total_steps} slots) ---')
for name in agent_names:
    m = metrics[name]
    rsr_arr = np.array(m["rsr"])
    cost_arr = np.array(m["cost"])
    delay_arr = np.array(m["t_all"])
    y_arr = np.array(m["y"])
    print(f'\n[{name}]')
    print(f'  rsr      = {rsr_arr.mean():.4f} ± {rsr_arr.std():.4f}  (min={rsr_arr.min():.4f}, max={rsr_arr.max():.4f})')
    print(f'  cost     = {cost_arr.mean():.2f} ± {cost_arr.std():.2f}  (C={cfg.C}, 利用率={cost_arr.mean()/cfg.C*100:.1f}%)')
    print(f'  t_all    = {delay_arr.mean():.3f} ± {delay_arr.std():.3f}  (T∈[{cfg.min_request_T},{cfg.max_request_T}])')
    print(f'  y        = {y_arr.mean():.2f} ± {y_arr.std():.2f}')
    print(f'  实例数   = {np.array(m["image_nums"]).mean():.1f}')
    print(f'  节点数   = {np.array(m["node_using_num"]).mean():.1f} / {cfg.node_nums}')
