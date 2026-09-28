# Stage2.3 History-Aware 5Q Critic Pretraining

Stage2.3 is an experimental five-Critic extension of the historical Stage2.2
contract used by `stage3v5_stage22_rnn4k_multi6k_20260922`.

It deliberately changes only the ensemble size:

- five independent recurrent Q networks;
- identical architecture per Q: token 64, LSTM hidden 96 / 1 layer, head 128;
- full-episode-prefix recurrent unroll from episode step zero;
- finite-episode Monte-Carlo return targets, no bootstrap;
- learning mask over 16 supervised transitions;
- effective 256 supervised timesteps per optimizer update;
- AdamW lr 3e-4, weight decay 1e-4;
- Multi source mix remains balanced BC-RNN / Transformer / GMM;
- RNN source remains BC-RNN only;
- 50,000 optimizer updates;
- held-out validation and checkpoint every 1,000 updates.

## Best checkpoint selection

The rule is fixed before training.  Across the 1K...50K validation checkpoints,
the selected checkpoint minimizes an equal normalized rank aggregate of:

- Spearman: maximize;
- Pearson: maximize;
- success/failure AUC: maximize;
- ensemble-mean MAE to finite MC return: minimize;
- ensemble-mean MSE to finite MC return: minimize.

For `rnn_q`, selection uses the held-out BC-RNN validation set.  For
`multi_q`, it uses the balanced aggregate over BC-RNN, BC-Transformer, and
BC-GMM.  This avoids selecting a checkpoint after looking at Stage3 results.

## Run

```bash
python training/Multi_IL_Full_Action_RL/stage2_3_history_aware_5q/train_stage2_3.py \
  --device npu:0 \
  --mode both
```

Outputs are written below
`training_runs/Multi_IL_Full_Action_RL/stage2_3_history_aware_5q/`.

The existing Stage2.2 code and checkpoints are not modified.
