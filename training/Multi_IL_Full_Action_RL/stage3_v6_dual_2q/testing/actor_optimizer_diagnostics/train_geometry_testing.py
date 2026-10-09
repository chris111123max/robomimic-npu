"""Testing-only real continuation for optimizer-geometry causality.

This wrapper derives the current production trainer at runtime and only:
  * swaps in GeometryAgent,
  * resumes from the formal random2q/multi_q critic_ready checkpoint,
  * writes exclusively under this testing directory,
  * adds milestones at 130K/140K/150K/160K,
  * marks every checkpoint testing-only.

No production file is edited.
"""
from __future__ import annotations

import hashlib
import inspect
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V6_DIR = HERE.parents[1]
if str(V6_DIR) not in sys.path:
    sys.path.insert(0, str(V6_DIR))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def main():
    import train_stage3_v6_vector as P
    import stage3_v6_agent as V6
    import geometry_agent as G
    import geometry_diagnostics as D

    branch = os.environ["OPT_GEOMETRY_BRANCH"]
    if branch not in G.BRANCHES:
        raise RuntimeError(branch)
    source_checkpoint = Path(
        os.environ["OPT_GEOMETRY_SOURCE_CHECKPOINT"]
    ).resolve()
    pair = Path(sys.argv[sys.argv.index("--quad-run-dir") + 1]).resolve()
    if not pair.is_relative_to(HERE):
        raise RuntimeError(f"Testing pair escaped testing directory: {pair}")
    if "--target-mode" not in sys.argv or sys.argv[sys.argv.index("--target-mode") + 1] != "random2q":
        raise RuntimeError("Geometry continuation is random2q-only")
    if "--group" not in sys.argv or sys.argv[sys.argv.index("--group") + 1] != "multi_q":
        raise RuntimeError("Geometry continuation is multi_q-only")
    total = int(sys.argv[sys.argv.index("--total-env-steps") + 1])
    if total != 160000:
        raise RuntimeError("Geometry real continuation is fixed to 160000 env steps")
    if "--resume" not in sys.argv:
        raise RuntimeError("Geometry continuation must resume from critic_ready")
    resume = Path(sys.argv[sys.argv.index("--resume") + 1]).resolve()
    if resume != source_checkpoint:
        raise RuntimeError(f"Unexpected resume checkpoint: {resume}")

    # Freeze source-code identity for this execution.
    production_files = [
        Path(P.__file__).resolve(),
        (V6_DIR / "stage3_v6_agent.py").resolve(),
        (V6_DIR.parent / "stage3_v5_rgmm_td3" / "stage3_v5_agent.py").resolve(),
        (V6_DIR / "stage3_v6_config.json").resolve(),
    ]
    source_hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in production_files
    }

    # The trainer imports this symbol locally inside main().
    V6.RecurrentGMMTD3 = G.GeometryAgent

    original_restore = P.restore_checkpoint
    def restore_checkpoint(path, agent, config, torch, OnlineSequenceReplay):
        payload, online = original_restore(
            path, agent, config, torch, OnlineSequenceReplay
        )
        agent.configure_after_resume(payload)
        return payload, online
    P.restore_checkpoint = restore_checkpoint

    original_checkpoint_payload = P.checkpoint_payload
    def checkpoint_payload(*args, **kwargs):
        payload = original_checkpoint_payload(*args, **kwargs)
        payload["testing_only"] = True
        payload["optimizer_geometry_branch"] = branch
        payload["optimizer_geometry_semantics"] = (
            "exact production Adam"
            if branch == "PRODUCTION_ADAM"
            else "global Adam step L2 matched, pure negative-gradient direction"
        )
        payload["production_source_sha256"] = source_hashes
        return payload
    P.checkpoint_payload = checkpoint_payload

    original_save = P.save_checkpoint
    def guarded_save(path, *args, **kwargs):
        resolved = Path(path).resolve()
        if not resolved.is_relative_to(HERE):
            raise RuntimeError(f"Testing checkpoint escaped testing root: {resolved}")
        return original_save(path, *args, **kwargs)
    P.save_checkpoint = guarded_save

    source = inspect.getsource(P.main)

    run_type_old = (
        'config["run_type"] = "BENCHMARK" if args.benchmark_mode '
        'else ("SMOKE" if args.smoke else "FORMAL")'
    )
    if source.count(run_type_old) != 1:
        raise RuntimeError("Production run_type source changed")
    source = source.replace(
        run_type_old,
        'config["run_type"] = "TESTING_ONLY_OPTIMIZER_GEOMETRY"',
        1,
    )

    checkpoint_old = (
        '        if args.benchmark_mode:\n'
        '            checkpoint_steps.add(int(args.benchmark_warmup_steps))'
    )
    if source.count(checkpoint_old) != 1:
        raise RuntimeError("Production checkpoint schedule source changed")
    source = source.replace(
        checkpoint_old,
        '        checkpoint_steps = {140000, 150000, 160000, total}\n'
        + checkpoint_old,
        1,
    )

    # Run the untouched 130K source Actor through the same diagnostic before
    # any resumed training update occurs.
    start_marker = "        metric_rows = []\n        active_cursor = 0"
    if source.count(start_marker) != 1:
        raise RuntimeError("Production metric-row marker changed")
    source = source.replace(
        start_marker,
        '        testing_milestone(agent, config, env_steps, group_dir, torch, '
        'source_checkpoint)\n'
        + start_marker,
        1,
    )

    checkpoint_call_tail = (
        '                                    episodes, successes, online, torch, '
        'handoff, credit.pending, offline)\n\n'
        '            # Reset all completed workers'
    )
    if source.count(checkpoint_call_tail) != 1:
        raise RuntimeError("Production in-loop checkpoint source changed")
    source = source.replace(
        checkpoint_call_tail,
        '                                    episodes, successes, online, torch, '
        'handoff, credit.pending, offline)\n'
        '                    testing_milestone(agent, config, env_steps, '
        'group_dir, torch, source_checkpoint)\n\n'
        '            # Reset all completed workers',
        1,
    )

    namespace = dict(vars(P))
    namespace["testing_milestone"] = D.testing_milestone
    namespace["source_checkpoint"] = source_checkpoint
    exec(
        compile(
            source,
            str(HERE / "derived_geometry_production_main"),
            "exec",
        ),
        namespace,
    )

    if "--validate-only" in sys.argv:
        print("GEOMETRY_DERIVED_PRODUCTION_MAIN_COMPILE_PASS", flush=True)
        return

    namespace["main"]()

    after_hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in production_files
    }
    if after_hashes != source_hashes:
        raise RuntimeError("Production source changed during testing continuation")
    print("GEOMETRY_TESTING_TRAINER_EXITED_ALL_VECTOR_ENVS_CLOSED", flush=True)
    print("FORMAL TRAINING REMAINS STOPPED", flush=True)


if __name__ == "__main__":
    main()
