# Training Overhead

## 10-node Twitter

Scenario: `twitter_largescale` (`node_nums=10`, `ms_nums=10`).

| Scenario | Method | Training Epochs | Convergence Epoch | Convergence Time (h) | Total Training Time (h) | SPS | Decision Mean (ms) | Decision P95 (ms) | Model Size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| twitter_largescale | AutoLFD (LGDRL) | 20000 | 11064 | 9.62 | 17.58 | 1452 | 1.027 | 1.067 | 2.51 MB |
| twitter_largescale | RL Agent | 4393 | 676 | 0.44 | 26.70 | 27 | 0.559 | 0.587 | 2.47 MB |
| twitter_largescale | DeepScaler | 10000 | 4701 | 8.84 | 18.98 | 668 | 1.513 | 1.562 | 3.63 MB |

Decision latency measures only `agent.get_action(state)` over 288 test slots; environment `step()` time is excluded.
SAC and DeepScaler use archived 10-node runs under `/home/zsw/Papers/AutoScaling_0619/AutoScaling/model/twitter_largescale/`.

## 20-node Twitter

Scenario: `twitter_xlargescale` (`node_nums=20`, `ms_nums=20`).

| Scenario | Method | Training Epochs | Convergence Epoch | Convergence Time (h) | Total Training Time (h) | SPS | Decision Mean (ms) | Decision P95 (ms) | Model Size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| twitter_xlargescale | AutoLFD (LGDRL) | 10000 | 8239 | 17.69 | 19.37 | 4659 | 1.227 | 1.101 | 3.37 MB |
| twitter_xlargescale | RL Agent | 7148 | 838 | 0.63 | 27.91 | 403 | 0.921 | 0.780 | 3.29 MB |
| twitter_xlargescale | DeepScaler | 10000 | 8888 | 20.61 | 21.67 | 6927 | 1.774 | 1.699 | 4.93 MB |

The 20-node rows use the `0619/2040` runs under `model/twitter_xlargescale/`, so AutoLFD keeps the same 3-hidden-layer architecture as the 10-node table; its model size is still larger because the state and action dimensions increase from 10x10 to 20x20.
Decision latency comes from `test_output/twitter_xlargescale/latency.npy`; SAC is stored there under the `RL Agent` key.
