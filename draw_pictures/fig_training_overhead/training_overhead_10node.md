# Training Overhead

## Paper Summary

This table reports AutoLFD convergence time and mean per-step decision time.

| Scenario | Training time (h) | Decision Time (ms) |
| --- | --- | --- |
| Sin | 5.10 | 0.4753 |
| Alibaba | 7.14 | 0.4958 |
| Twitter | 9.62 | 0.5107 |
| Twitter(20) | 17.69 | 0.4931 |

Training time is the convergence time detected from `charts/y`; decision time is mean per-step `agent.get_action(state)` measured locally on CPU (100 episodes × 288 steps = 28,800 samples, warmup=30).


## 10-node Twitter

Scenario: `twitter_largescale` (`node_nums=10`, `ms_nums=10`).

| Scenario | Method | Training Epochs | Convergence Epoch | Convergence Time (h) | Total Training Time (h) | SPS | Decision Mean (ms) | Decision P95 (ms) | Model Size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| twitter_largescale | AutoLFD (LGDRL) | 20000 | 11064 | 9.62 | 17.58 | 1452 | 1.715 | 4.447 | 2.51 MB |
| twitter_largescale | SAC | 4393 | 676 | 0.44 | 26.70 | 27 | 2.717 | 3.427 | 2.47 MB |
| twitter_largescale | DeepScaler | 10000 | 4701 | 8.84 | 18.98 | 668 | 2.087 | 3.681 | 3.63 MB |

Decision latency measures only `agent.get_action(state)` over 288 test slots; environment `step()` time is excluded.
SAC and DeepScaler use archived 10-node runs under `/home/zsw/Papers/AutoScaling_0619/AutoScaling/model/twitter_largescale/`; decision latency comes from the matching archived `test_output/twitter_largescale/latency.npy`.

## 20-node Twitter

Scenario: `twitter_xlargescale` (`node_nums=20`, `ms_nums=20`).

| Scenario | Method | Training Epochs | Convergence Epoch | Convergence Time (h) | Total Training Time (h) | SPS | Decision Mean (ms) | Decision P95 (ms) | Model Size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| twitter_xlargescale | AutoLFD (LGDRL) | 10000 | 8239 | 17.69 | 19.37 | 4659 | 1.227 | 1.101 | 3.37 MB |
| twitter_xlargescale | SAC | 7148 | 838 | 0.63 | 27.91 | 403 | 0.921 | 0.780 | 3.29 MB |
| twitter_xlargescale | DeepScaler | 10000 | 8888 | 20.61 | 21.67 | 6927 | 1.774 | 1.699 | 4.93 MB |

The 20-node rows use the `0619/2040` runs under `model/twitter_xlargescale/`, so AutoLFD keeps the same 3-hidden-layer architecture as the 10-node table; its model size is still larger because the state and action dimensions increase from 10x10 to 20x20.
Decision latency comes from `test_output/twitter_xlargescale/latency.npy`; SAC is stored there under the `RL Agent` key.
