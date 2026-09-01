# Upstream provenance

- Repository: `git@github.com:twni2016/pomdp-baselines.git`
- Upstream commit: `e7c19c32a20033d75414b29fbc466c77c211e968`
- License: MIT; see `LICENSE`.
- Integration: vendored source dependency (not a Git submodule).

Local change: `torchkit/pytorch_utils.py` adds a generic `set_device` helper so
ordinary PyTorch modules and distribution utilities can use `npu:0`. SAC loss,
entropy handling, target updates, recurrent policies, and replay buffers are
unchanged.

The legacy `future_fstrings` source-code encoding declarations were replaced
with standard UTF-8 declarations. Python 3.10 natively supports the f-string
syntax used by these files, so the external codec is unnecessary.
