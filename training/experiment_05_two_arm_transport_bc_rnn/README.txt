Experiment 05 — TwoArmTransport PH low_dim BC-RNN

Purpose
-------
Reproduce the robomimic paper-style BC-RNN baseline on:
- task: Transport / TwoArmTransport
- dataset: PH
- observation: low_dim_v15

Official BC-RNN settings retained
---------------------------------
- algo_name = bc
- train.seq_length = 10
- algo.rnn.enabled = true
- algo.rnn.horizon = 10
- algo.rnn.hidden_dim = 400
- algo.rnn.rnn_type = LSTM
- algo.rnn.num_layers = 2
- algo.rnn.open_loop = false
- algo.actor_layer_dims = []
- algo.gmm.enabled = true (PH human demonstrations)
- GMM modes = 5
- policy learning rate = 1e-4
- batch_size = 100
- num_epochs = 2000
- 100 gradient steps / training epoch
- 10 validation steps / validation epoch
- rollout every 50 epochs
- 50 rollout episodes per evaluation
- Transport-PH rollout horizon = 700
- terminate_on_success = true
- save every 50 epochs
- save on best rollout success rate

Local-only adaptations
----------------------
Only experiment-local information was adapted:
1. dataset path -> /data/home/3220251075/lerobot_workspace/datasets/transport/PH/low_dim_v15.hdf5
2. output path -> /data/home/3220251075/lerobot_workspace/training_runs/two_arm_transport_bc_rnn_official_ph_low_dim
3. experiment name
4. TwoArmTransport low-dimensional observation list includes both Panda arms, matching the official Transport config logic and the local v1.5 dataset.
5. train.data remains a string path to match the local robomimic configuration format already verified on this machine.

Run
---
Unzip/copy this folder to:
  /data/home/3220251075/lerobot_workspace/training/experiment_05_two_arm_transport_bc_rnn

Then:
  cd /data/home/3220251075/lerobot_workspace/training/experiment_05_two_arm_transport_bc_rnn
  NPU_ID=2 ./run.sh

Or, if the conda + CANN environment is already active:
  python launch.py

Important
---------
This is the paper-style BC-RNN for PH data, which is actually BC-RNN-GMM:
RNN enabled + GMM enabled.

The config deliberately uses the official evaluation protocol (50 rollouts every 50 epochs),
not the previous reduced 10-rollout protocol.
