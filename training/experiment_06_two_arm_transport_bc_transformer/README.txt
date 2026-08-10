Experiment 06 — TwoArmTransport PH low_dim BC-Transformer-GMM
================================================================

Design principle
----------------
Use robomimic official settings wherever an official setting exists.

Transformer-specific settings are taken from the robomimic tuned default:
  robomimic/config/default_templates/bc_transformer.json

Transport / PH / low_dim dataset handling and rollout protocol follow
robomimic's official low-dimensional experiment conventions where applicable.

Core algorithm
--------------
Actual class selected:
  BC_Transformer_GMM

because:
  algo.transformer.enabled = true
  algo.gmm.enabled         = true
  algo.rnn.enabled         = false

Official tuned Transformer settings retained
--------------------------------------------
train.seq_length             = 1
train.frame_stack            = 10
batch_size                   = 100
num_epochs                   = 2000

optimizer                    = AdamW
learning_rate                = 1e-4
scheduler_type               = linear
epoch_schedule               = [100]
L2 / weight decay            = 0.01

actor_layer_dims             = []
GMM enabled                  = true
GMM modes                    = 5

Transformer:
  enabled                    = true
  context_length             = 10
  embed_dim                  = 512
  num_layers                 = 6
  num_heads                  = 8
  emb_dropout                = 0.1
  attn_dropout               = 0.1
  block_output_dropout       = 0.1
  activation                 = gelu
  supervise_all_steps        = false
  pred_future_acs            = false

Evaluation / dataset conventions
--------------------------------
task                         = TwoArmTransport
dataset                      = PH low_dim_v15
train filter                 = train
validation filter            = valid
validation                   = enabled
rollout horizon              = 700
rollouts per evaluation      = 50
rollout every                = 50 epochs
gradient steps / epoch       = 100
validation steps / epoch     = 10
terminate_on_success         = true

Local adaptations only
----------------------
1. Dataset path points to the local v1.5 Transport PH low_dim dataset.
2. Output directory and experiment name are local.
3. Transport observations use both Panda arms plus object state.
4. Train/valid masks are used so Exp6 trains on the same demonstrations
   as the other PH experiments rather than silently using all 200 demos.
5. Local action schema fields are retained for compatibility with the
   installed robomimic tree.
6. run.sh defaults to NPU 3; this is only a launcher choice.

Run
---
Copy / unzip to:
  /data/home/3220251075/lerobot_workspace/training/experiment_06_two_arm_transport_bc_transformer

Foreground:
  cd /data/home/3220251075/lerobot_workspace/training/experiment_06_two_arm_transport_bc_transformer
  NPU_ID=3 ./run.sh

nohup:
  cd /data/home/3220251075/lerobot_workspace/training/experiment_06_two_arm_transport_bc_transformer
  nohup env NPU_ID=3 bash run.sh > nohup_bc_transformer.log 2>&1 &
  echo $!

Watch:
  tail -f nohup_bc_transformer.log

Expected model family in startup output
---------------------------------------
BC_Transformer_GMM
  -> TransformerGMMActorNetwork
  -> context 10
  -> 6 transformer layers
  -> embedding dim 512
  -> 8 attention heads
  -> 5-mode GMM action head

Important difference from BC-RNN
--------------------------------
BC-RNN:
  seq_length = 10
  frame_stack = 1

BC-Transformer official tuned template:
  seq_length = 1
  frame_stack = 10

Do not change these two fields just to make the configs look similar.
