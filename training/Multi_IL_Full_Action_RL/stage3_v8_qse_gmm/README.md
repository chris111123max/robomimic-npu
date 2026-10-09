# Stage3-V8 QSE-GMM

Q-Guided Successful Experience GMM Fine-Tuning. This directory is isolated from V5/V6/V7. Formal training is NOT started by this implementation task.

## Read first

- V8_GITHUB_RESEARCH.md: six pinned repositories and actual source decisions.
- V8_ALGORITHM_DESIGN.md: objective, verified-success data, calibration, compatibility and future acceptance.
- FINAL_REPORT.md: results and limits.
- testing/test_results.json and testing/calibration.json: numerical evidence.

## Safe preparation and CPU tests

Activate the existing robosuite_npu environment on bit-robomimic and run from the authoritative repository:

```bash
export PYTHONDONTWRITEBYTECODE=1
python training/Multi_IL_Full_Action_RL/stage3_v8_qse_gmm/train_stage3_v8_vector.py
python training/Multi_IL_Full_Action_RL/stage3_v8_qse_gmm/testing/test_cpu.py
```

The first command performs read-only preflight and returns PREPARED / STOPPED. It creates no simulator and takes no optimizer step. The test suite uses CPU clones and a nonzero-LR Adam step solely to test equivalence. Test migration checkpoints are written only under V8/testing/checkpoint_validation, never into the V7 run.

The recorded calibration is fixed in stage3_v8_config_calibrated.json. stage3_v8_config.json is an intentionally unresolved template. Do not re-calibrate automatically during resume or use a different V7 checkpoint. Re-running tests creates a new immutable test checkpoint and updates test summaries; previous binary checkpoints remain intact.

## Later formal training (not authorized or executed here)

Only after explicit subsequent authorization, the same entry point supports --execute. It requires npu:0, 16 original CPU simulator workers, original readiness and LR schedule, random2q/multi_q and the exact V7 340K fork. A V8 continuation must use --resume pointing to its own immutable .pth and .sequences.npy bundle. LATEST.json is a pointer to the most recent checkpoint; no old binary is overwritten.

The loop's output directory is:
training_runs/Multi_IL_Full_Action_RL/stage3_v8_qse_gmm/<prepared-run-name>/random2q/multi_q/

The adapter checks the audited V7 trainer SHA256 and four exact extension points before running. If upstream V7 source changes, re-audit and update the adapter contract deliberately. It does not change production source files or patch production module globals.

## Files

- stage3_v8_agent.py: additive success gradient before original V7 clipping/Adam. lambda_good=0 directly delegates to V7.
- stage3_v8_gmm_loss.py: five-component, 14-dimensional normalized-action GMM logsumexp likelihood and weighted mask reduction.
- stage3_v8_good_replay.py: independent outcome-filtered pool/RNG; inherited ordinary critic replay samplers unchanged.
- stage3_v8_checkpoint.py: exact 340K fork, strict V8 resume, full pool/RNG/optimizer/targets/handoff state and replay digest.
- train_stage3_v8_vector.py: fail-closed V7-loop adapter and safe preflight.
- testing/calibrate.py: bounded offline calibration; not part of runtime adaptation.
- testing/test_cpu.py: required offline tests plus adapter preflight.

Success labels are conservative. Offline historical success-collector stops can have truncated=True; accept only their explicit audited success reason and full terminal subtask corroboration. Online timeout, unfinished episode, failed task or one-subtask completion is excluded. Full ten-step aligned windows omit incomplete episode tails; no padding is used in replay.

Offline tests establish implementation correctness, not improved closed-loop success or NPU training performance. No V8 training, no new online evaluation, no GitHub push occurred.
