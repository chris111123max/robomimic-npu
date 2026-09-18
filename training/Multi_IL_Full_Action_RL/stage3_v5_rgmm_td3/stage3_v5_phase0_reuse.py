"""Compatibility stub: Stage3-v5 deliberately does not reuse Phase-0 gates."""


def validate_phase0_reuse(*_args, **_kwargs):
    raise RuntimeError(
        "Stage3-v5 does not reuse V3/V4 Phase-0 artifacts; readiness is replay-only"
    )
