""" 20节点 LGDRL「4层+20000epoch」容量实验 —— eval + 对比绘图。

eval LGDRL-4L(deep) vs DeepScaler vs SAC vs HPA，在 twitter / alibaba 20节点上各跑 288 步，
叠加 twitter 原 3 层 LGDRL 的冻结数据，画 delay/cost/RSR 分组柱状图 + 存 JSON。输出到 draw_pictures/fig4/。

注意：
 1) PPO_dnn.py 当前是 3 层，但 _deep 模型是 4 层训练的 → 用临时 4 层 ActorCritic 加载（不改 PPO_dnn.py）。
 2) SAC.Actor 用模块级 CONFIG → 必须在 import env 前把 AUTOSCALING_CONFIG 设成 20节点 config。
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

REPO = "/home/zsw/Papers/AutoScaling"
sys.path.insert(0, REPO)
os.chdir(REPO)
# 必须在 import env 前设：SAC.Actor / DeepScaler 用模块级 environment.CONFIG 建网。
# twitter_xlargescale_deep 是 20节点，与 alibaba_xlargescale_deep 维度完全一致，故一份即可。
os.environ["AUTOSCALING_CONFIG"] = "twitter_xlargescale_deep"

from env import environment
from methods import PPO_dnn, DeepScaler, SAC, HPA, ProScaling
from env.configs.config_twitter_xlargescale_deep import EnvConfig as TwCfg
from env.configs.config_alibaba_xlargescale_deep import EnvConfig as AliCfg

MODELS = {
    "twitter": {
        "cfg": TwCfg(),
        "lg_deep": "model/twitter_xlargescale_deep/0621/1617/PPO_dnn/model_dnn_best.pth",
        "sac": "model/twitter_xlargescale/0619/2040/SAC",
        "ds": "model/twitter_xlargescale/0619/2040/DeepScaler/model_1000.pth",
    },
    "alibaba": {
        "cfg": AliCfg(),
        "lg_deep": "model/alibaba_xlargescale_deep/0621/1617/PPO_dnn/model_dnn_best.pth",
        "sac": "model/alibaba_xlargescale/0621/0237/SAC",
        "ds": "model/alibaba_xlargescale/0621/0237/DeepScaler/model_1000.pth",
    },
}
FROZEN_TW_LG3 = "test_output/twitter_xlargescale/data/LGDRL"  # 原 3 层 LGDRL，无法重跑（arch 变了）


def load_deep_ppo(cfg, path):
    """PPO 模型层数可能与当前 PPO_dnn.py 不一致（用户在别的实验里改层数）。
    从 checkpoint 的 state_dict 自动探测 dnn 的 Linear 层数，构造等深 dnn 再加载。"""
    sd = torch.load(path, weights_only=True, map_location="cpu")
    dnn_idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith("dnn.") and k.endswith(".weight")})
    n_layers = len(dnn_idx)
    ag = PPO_dnn.PPOAgent(config=cfg)
    inp = int(np.sum(ag.actorcrtic.feature_length_list))
    layers = [PPO_dnn.layer_init(nn.Linear(inp, 512)), nn.ReLU()]
    for _ in range(n_layers - 1):
        layers += [PPO_dnn.layer_init(nn.Linear(512, 512)), nn.ReLU()]
    ag.actorcrtic.dnn = nn.Sequential(*layers).to(cfg.device)
    ag.actorcrtic.load_state_dict(sd)
    print(f"  [load] {os.path.basename(path)}: 探测到 {n_layers} 层 MLP", flush=True)
    return ag


def eval_agent(agent, env, cfg):
    """跑满一个 episode（288 步），返回 per-step 数组。
    注意：HPA/ProScaling 的 get_action 用 self.env 内部状态，故必须 step 它们持有的那个 env。"""
    obs = env.reset(seed=1037)[0]
    done = False
    t_all, cost, rsr = [], [], []
    while not done:
        a = agent.get_action(obs)
        obs, _, done, _, info = env.step(a)
        t_all.append(np.mean(info["t_all"]))
        cost.append(np.mean(info["cost"]))
        rsr.append(np.mean(info["request_success_rate"]))
    return np.array(t_all), np.array(cost), np.array(rsr)


def build_agents(cfg, m):
    """返回 {名: (agent, 要 step 的 env)}。启发式 agent 持有的 env == 要 step 的 env。"""
    temp = environment.DataCenterEnvironment(0, cfg, is_train=True)
    temp.reset(seed=cfg.seed)
    lg = load_deep_ppo(cfg, m["lg_deep"])
    ds = DeepScaler.DeepScalerAgent(cfg, temp.MS2MS_data_graph); ds.load(m["ds"])
    sac = SAC.SACAgent(config=cfg); sac.load(m["sac"])
    env_hpa = environment.DataCenterEnvironment(0, cfg); hpa = HPA.HPA(env_hpa)
    env_pro = environment.DataCenterEnvironment(0, cfg); pro = ProScaling.ProScalingAgent(env_pro)
    # 内部名 → fig4 目录名（draw_xlarge.py 读取的 key）；启发式配对持有的 env，RL 配对新 env
    return {
        "HPA": (hpa, env_hpa),
        "Proscale": (pro, env_pro),
        "RL Agent": (sac, environment.DataCenterEnvironment(0, cfg)),
        "DeepScaler": (ds, environment.DataCenterEnvironment(0, cfg)),
        "LGDRL": (lg, environment.DataCenterEnvironment(0, cfg)),
    }


# ── eval ──────────────────────────────────────────────────────────────────────
results = {"twitter": {}, "alibaba": {}}
for ds_name in ["twitter", "alibaba"]:
    m = MODELS[ds_name]
    cfg = m["cfg"]
    print(f"=== eval {ds_name} 20节点 ===", flush=True)
    for name, (ag, env) in build_agents(cfg, m).items():
        t_arr, c_arr, r_arr = eval_agent(ag, env, cfg)
        results[ds_name][name] = {"delay": float(t_arr.mean()), "cost": float(c_arr.mean()), "rsr": float(r_arr.mean())}
        print(f"  {name:12s} delay={t_arr.mean():6.2f}  cost={c_arr.mean():6.2f}  RSR={r_arr.mean():.3f}", flush=True)
        # dump per-step npy 到 test_output/{config}/data/{agent}/，供 draw_xlarge.py 画逐时隙图
        out_data = os.path.join(REPO, "test_output", cfg.config_name, "data", name)
        os.makedirs(out_data, exist_ok=True)
        np.save(os.path.join(out_data, "t_all.npy"), t_arr)
        np.save(os.path.join(out_data, "cost.npy"), c_arr)
        np.save(os.path.join(out_data, "request_success_rate.npy"), r_arr)
    # twitter 的原 3 层 LGDRL 冻结数据（紧跟 twitter 块，避免被误读成 alibaba）
    if ds_name == "twitter" and os.path.isdir(FROZEN_TW_LG3):
        fd = FROZEN_TW_LG3
        results["twitter"]["LGDRL-3L"] = {
            "delay": float(np.load(fd + "/t_all.npy").mean()),
            "cost": float(np.load(fd + "/cost.npy").mean()),
            "rsr": float(np.load(fd + "/request_success_rate.npy").mean()),
        }
        lg3 = results["twitter"]["LGDRL-3L"]
        print(f"  LGDRL-3L(冻结,twitter原3层) delay={lg3['delay']:.2f}  cost={lg3['cost']:.2f}  RSR={lg3['rsr']:.3f}", flush=True)

out_dir = os.path.join(REPO, "draw_pictures/fig4")
os.makedirs(os.path.join(out_dir, "data"), exist_ok=True)
with open(os.path.join(out_dir, "data", "xlarge_deep_comparison.json"), "w") as f:
    json.dump(results, f, indent=2)

# ── 绘图：3 子图 (delay/cost/RSR)，按数据集分组，组内各方法柱 ─────────────────
methods = ["HPA", "Proscale", "RL Agent", "DeepScaler", "LGDRL-3L", "LGDRL"]
colors = {"HPA": "#376eae", "Proscale": "#f8c91c", "RL Agent": "#449945",
          "DeepScaler": "#8E6EC8", "LGDRL-3L": "#e8a0a0", "LGDRL": "#c22f2f"}
datasets = ["twitter", "alibaba"]
metrics = [("delay", "Delay"), ("cost", "Cost"), ("rsr", "RSR")]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
x = np.arange(len(datasets))
width = 0.16
n_methods = len(methods)
for ax, (mk, mlabel) in zip(axes, metrics):
    for i, meth in enumerate(methods):
        vals = [results[dsname].get(meth, {}).get(mk, np.nan) for dsname in datasets]
        offset = (i - (n_methods - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=meth, color=colors[meth],
                      edgecolor="black", linewidth=0.4)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in datasets], fontsize=13)
    ax.set_ylabel(mlabel, fontsize=14)
    ax.set_title(mlabel, fontsize=15)
    ax.grid(True, axis="y", alpha=0.3)
axes[-1].legend(fontsize=11, loc="upper right")
fig.suptitle("20-node: LGDRL 4-layer (20000 epochs) vs baselines", fontsize=16)
plt.tight_layout()
pdf = os.path.join(out_dir, "xlarge_deep_comparison.pdf")
plt.savefig(pdf, format="pdf")
plt.close()
print(f"\nPDF  → {pdf}")
print(f"JSON → {out_dir}/xlarge_deep_comparison.json")
