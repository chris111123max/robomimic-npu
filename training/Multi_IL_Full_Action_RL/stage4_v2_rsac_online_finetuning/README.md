# Stage4-v2: Protected Low-Entropy Recurrent SAC

This is an isolated 100,000-transition mechanism experiment for the RNN-only
and Multi-IL Critic initializations. It reuses the Stage4-v1 recurrent SAC,
sequence replay, parallel robosuite pool, NPU adapter, evaluation, checkpoint,
resume, and numerical-failure protection.

The only optimization changes are a 5,000-step Actor freeze, linear Actor LR
warmup from step 5,000 to 20,000, and fixed entropy alpha 0.001 with automatic
entropy tuning disabled. Each group uses eight CPU simulation workers. Random
Critic, demonstrations, BC/KL losses, random actions, and automatic extension
to one million transitions are not used.

Run preflight only:

```bash
bash training/Multi_IL_Full_Action_RL/stage4_v2_rsac_online_finetuning/run_stage4_v2_two_groups.sh smoke
```

Start the two formal 100k jobs:

```bash
bash training/Multi_IL_Full_Action_RL/stage4_v2_rsac_online_finetuning/run_stage4_v2_two_groups.sh formal
```

The launcher uses logical NPU 1/2 when at least three devices are visible, and
falls back to logical NPU 0/1 when the container exposes exactly two devices.
