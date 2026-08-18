# 实验二：局部能力验证（Local Competence / Branch Takeover）

## 研究问题

Experiment 00 已经在相同的 100 个 initial conditions 上证明：由同一 demonstration dataset 训练出的不同 IL policy，其 episode-level 成功集合不完全重合。它仍不能说明两个 policy 在同一条真实轨迹的同一中间条件下是否具有不同的后续完成能力。

本实验只比较 `bc_gmm_rnn` 与 `bc_gmm_transformer`。它在 source policy 的真实失败轨迹上保存中间条件

`x_t = (完整 simulator state_t, trajectory history H_t)`，

再从同一个 `x_t` 分别估计 source continuation 与 target takeover 的成功率。本实验不训练 policy、不修改 checkpoint、不做 RL、SAC、critic、replay buffer、gating 或 action ensemble。

## 固定术语

- **source failure trajectory**：source policy 从 Experiment 00 的固定 initial state 出发、实际执行并最终失败的一条轨迹。
- **branch point**：执行 `a_t` 之前的决策点。此时已经完成 `a_0...a_{t-1}`，环境位于 `state_t`，持有 `obs_t`。
- **source continuation**：source policy 从相同 `x_t` 继续。
- **target takeover**：另一个 policy 从相同 `x_t` 接管。
- **local competence gap**：`target_success_rate - source_success_rate`。
- **strong local rescue**：预注册为 target rate `>= 2/3` 且 source rate `<= 1/3`。
- **rescue window**：同一 episode 中 strong rescue 的最早、最晚 branch step 及可救 branch point 数量。

## 为什么不能只恢复 simulator state

RNN 的 action 依赖内部循环隐藏状态；Transformer checkpoint 使用与 context length 一致的 frame stack。仅执行 `reset_to(state_t)` 后调用 `start_episode(); policy(obs_t)` 会删除真实历史，得到的不是原来的决策条件。

本实现不猜测 LSTM tensor shape，也不手工拼 Transformer token。它复用 Experiment 00 的官方 `RolloutPolicy` 与 frame-stack 包装语义：

1. `policy.start_episode()`；
2. 顺序喂入 source trajectory 的 `obs_0...obs_{t-1}`，忽略且绝不执行 warm-up actions；
3. 恢复 captured RNG（重建检查）或设置明确的 branch trial seed（正式分支）；
4. 第一次真实分支调用只喂一次 `obs_t`，因此没有 double-feed。

当前 repo 的 `BC_RNN_GMM.get_action` 和 `BC_Transformer_GMM.get_action` 只接收 observation / optional goal，不接收 previous action。Transformer 历史由 observation frame stack 表示；RNN 历史由逐 observation 前向恢复。若未来 checkpoint 的真实接口引入 action history，validate 会拒绝继续，而不是使用 target policy 未执行的假 action。

## 实验流程

1. `validate`：核对 source run、100 个状态、seed 映射、固定统计、dataset/checkpoint、环境、action/observation interface、最小 inference 与四个 NPU mask。
2. `select`：自动选出全部 16 个 RNN-fail/T-success 和 42 个 T-fail/RNN-success case；仅 smoke 可用 `--max-cases`，按每个方向确定性取前 N 个。
3. `recheck`：两个 policy 各重复 3 次。物理 initial state 与环境 RNG 固定，只改变 Torch/NPU policy seed。全部 raw result 保留，稳定性阈值来自 config。
4. `build`：优先重放最小失败 recheck trial 的 seed；否则最多搜索 10 个新 seed。只在真实复现失败后保存 source trajectory 和 branch states。
5. `reconstruct`：用官方 `get_state/reset_to` 检查完整 state hash，重放历史，并在可恢复 accelerator RNG 时要求原 action 与 reconstructed action 在 `atol=1e-5` 内一致。若当前 torch_npu 缺少 RNG state API，明确标记 `unavailable_exact_rng`，仍强制 state/hash/history 正确；对三个显式 seed 各完整重建两次，要求同 seed action 可重复、数值有限、形状正确且位于 action bounds 内。任何真正的重建失败都会阻止 branch stage。
6. `branch`：每个 branch condition、每个 policy 各 3 个明确 seed；总 task step 不超过 700；success 使用 `env.is_success()["task"]`。source policy 必须重跑，因为原失败只是一条 sample，实验需要估计 `P(source succeeds | x_t)`。
7. `analyze`：生成 direction、step、episode rescue window、continuous gap 和 strong rescue 统计；只报告数据，不自动宣布研究假设成立。

每个 branch 重复 3 次，是为了把 GMM 的单次随机 0/1 outcome 转成受控的经验成功率，同时控制第一轮正式实验的成本。后续可用 `--branch-repeats 5` 新建独立 run，但不得覆盖本轮预注册结果。

## 状态与 RNG 语义

branch state 保存官方 `EnvRobosuite.get_state()` 返回的完整 `model`、`states` 和 `ep_meta`，并保存 `obs_t`、原始 `action_t`、state/observation hash 与 branch-action 前 RNG。恢复使用官方 `reset_to()`，恢复后的完整 hash、state-vector hash 和 observation hash 必须一致；mismatch 是 runtime error，不是 task failure。

环境 RNG 与 policy RNG 分离。每次 `reset_to` 前设置固定的环境 Python/NumPy stream；环境恢复完成后只设置 Torch CPU/NPU policy stream，因此不同 branch repeat 不会重新随机物理环境。

## 并行与续跑

`recheck`、`build`、`reconstruct` 和 `branch` 默认都使用四个独立 worker，并分别绑定 config 中的四个 NPU mask。前三个阶段按 initial-state case 确定性分片；每个 worker 写自己的 CSV，全部 worker 正常退出后才由主进程原子聚合正式 CSV。这样既不并发改写同一个表，也可以复用旧版单进程已经写入的正式 CSV。

branch stage 的策略分片为：

- worker 0 / mask 0：RNN，shard 0；
- worker 1 / mask 1：Transformer，shard 0；
- worker 2 / mask 2：RNN，shard 1；
- worker 3 / mask 3：Transformer，shard 1。

mask 是 `ASCEND_RT_VISIBLE_DEVICES` 的可见设备 mask，不假定 `npu-smi` 的物理编号为 0–3。每个 worker 在自己的进程中创建环境并加载 checkpoint；主进程结束后统一聚合，避免并发 append 同一 CSV。

同一 `--run-dir` 自动 resume。recheck 的唯一键为 `(initial_state_id, policy_name, trial_index)`，build 为 `initial_state_id`，reconstruct 为 `(initial_state_id, branch_step)`，branch 为 `(initial_state_id, direction, branch_step, evaluated_policy, trial_index)`。worker 启动时同时读取正式聚合表和自己的分片表，因此中断只会重跑尚未完整落盘的当前任务。重要 JSON/CSV/NPZ 使用临时文件、flush/fsync 和原子 replace。可分别使用：

`--force-recheck`、`--force-source-trajectories`、`--force-reconstruction`、`--force-branch-eval`、`--force-analysis`。

四卡阶段的实时日志位于 `logs/<stage>_worker_<id>.log`；worker 分配表位于对应阶段的 `workers/worker_assignment.json`。recheck 运行期间的新增进度先写入 `recheck/workers/worker_<id>_trials.csv`，阶段完成后才合并回 `recheck/raw_trials.csv`。

已完成 validate/select 的正式 run 可以用 `--stage resume --run-dir <原目录> --config <原目录>/config.json` 从 recheck 自动续跑到最终 analyze；它会校验并复用正式表与各 worker 分片，不需要手工指定已经完成多少条。

## 输出

输出位于 `training_runs/Heterogeneous_IL_RL/experiment_02_local_competence/<timestamp>/`，包含固定 source manifest、selected cases、recheck raw/case summary、source trajectories、branch states、reconstruction gate、四 worker raw files、aggregate tables、analysis JSON/CSV 与两张 PNG 图。只保存 low-dimensional observation，不保存 RGB 或视频。

## 服务器运行

所有命令都从 robomimic 根目录执行。推荐先单独 validate，再用新的 timestamp 做 smoke，最后再启动 formal。不要把 smoke 的 `--run-dir` 用于 formal。

本地开发环境没有 Ascend、robosuite runtime、dataset、checkpoint 和 Experiment 00 输出，因此源码侧只能完成语法、CLI 和纯数据逻辑检查。真实服务器仍必须验证：CANN RNG state API、完整 XML state hash 在 `reset_to` 后是否字节稳定、两种 policy 的 history reconstruction、四卡 mask 和真实 rollout 数量。程序会对这些条件 fail fast，不会生成或伪造 rollout 数字。
