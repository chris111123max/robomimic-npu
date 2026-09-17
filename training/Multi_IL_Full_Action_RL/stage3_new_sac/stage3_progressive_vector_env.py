"""Spawn-based staggered MuJoCo workers for the progressive experiment only.

The training process owns all torch / NPU models.  Workers own only one
robosuite / MuJoCo environment each.  Workers are started sequentially and a
worker must report READY before the next worker is started.  This avoids the
large initialization spike that previously crashed the terminal / job.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}


def _worker_cpu_affinity(env_id: int) -> None:
    """Optionally pin one MuJoCo worker to one CPU core.

    This is an execution-only optimization.  It is disabled unless the
    parent supplies ``STAGE3_ENV_CPU_CORES`` as a comma-separated list or
    range (for example ``0-15,32-35``), so existing runs are unchanged.
    """
    specification = os.environ.get("STAGE3_ENV_CPU_CORES", "").strip()
    if not specification or not hasattr(os, "sched_setaffinity"):
        return
    cores = []
    try:
        for token in specification.split(","):
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                first, last = (int(value) for value in token.split("-", 1))
                cores.extend(range(first, last + 1))
            else:
                cores.append(int(token))
        cores = sorted(set(cores))
        if not cores:
            return
        os.sched_setaffinity(0, {cores[int(env_id) % len(cores)]})
    except (OSError, ValueError):
        # Cgroups may expose only a subset of the requested host cores.  A
        # failed optional pin must never abort training.
        return


def _safe_send(conn, message) -> None:
    try:
        conn.send(message)
    except (BrokenPipeError, EOFError, OSError):
        pass


def _action_bounds(env):
    """Read robosuite action bounds through robomimic wrapper layers."""
    current = env
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        spec = getattr(current, "action_spec", None)
        if spec is not None:
            low, high = spec
            return (
                np.asarray(low, dtype=np.float32),
                np.asarray(high, dtype=np.float32),
            )
        current = getattr(current, "env", None)
    raise RuntimeError(
        "Cannot obtain robosuite action_spec through environment wrappers"
    )


def _worker(conn, env_id: int, dataset: str, initial_seed: int) -> None:
    """Own exactly one CPU MuJoCo environment."""
    # These values are also injected by the parent before spawn so that they
    # are already visible while the child imports numpy / MuJoCo.
    os.environ.update(_THREAD_ENV)
    _worker_cpu_affinity(env_id)

    env = None
    try:
        from stage3_new_evaluation import (
            build_env,
            close_env,
            mujoco_fatal_error_type,
            reset_seed,
            success,
        )

        env = build_env(dataset)
        obs = reset_seed(env, int(initial_seed))
        low, high = _action_bounds(env)
        _safe_send(
            conn,
            (
                "READY",
                obs,
                np.asarray(low, dtype=np.float32),
                np.asarray(high, dtype=np.float32),
            ),
        )

        while True:
            try:
                command, payload = conn.recv()
            except EOFError:
                break

            if command == "step":
                try:
                    obs, reward, done, info = env.step(payload)
                    _safe_send(
                        conn,
                        (
                            "OK",
                            obs,
                            float(reward),
                            bool(done),
                            bool(success(env)),
                            info,
                        ),
                    )
                except mujoco_fatal_error_type() as error:
                    # The worker stays alive.  The parent can request a full
                    # rebuild for this env only.
                    _safe_send(conn, ("FATAL", str(error)))

            elif command == "reset":
                try:
                    obs = reset_seed(env, int(payload))
                    _safe_send(conn, ("RESET_OK", obs))
                except mujoco_fatal_error_type() as error:
                    _safe_send(conn, ("RESET_FATAL", str(error)))

            elif command == "rebuild":
                seed = int(payload)
                try:
                    if env is not None:
                        close_env(env)
                    env = build_env(dataset)
                    obs = reset_seed(env, seed)
                    low, high = _action_bounds(env)
                    _safe_send(
                        conn,
                        (
                            "REBUILD_OK",
                            obs,
                            np.asarray(low, dtype=np.float32),
                            np.asarray(high, dtype=np.float32),
                        ),
                    )
                except BaseException as error:
                    _safe_send(
                        conn,
                        (
                            "ERROR",
                            env_id,
                            repr(error),
                            traceback.format_exc(),
                        ),
                    )
                    break

            elif command == "close":
                _safe_send(conn, ("CLOSED",))
                break

            else:
                raise RuntimeError(f"Unknown worker command {command!r}")

    except BaseException as error:
        _safe_send(
            conn,
            (
                "ERROR",
                env_id,
                repr(error),
                traceback.format_exc(),
            ),
        )
    finally:
        if env is not None:
            try:
                from stage3_new_evaluation import close_env

                close_env(env)
            except BaseException:
                pass
        try:
            conn.close()
        except BaseException:
            pass


class StaggeredVectorEnv:
    """Small spawn-vector backend dedicated to Stage3 progressive training."""

    def __init__(
        self,
        dataset: str,
        num_envs: int,
        seed_base: int,
        delay: float = 0.5,
        timeout: float = 120.0,
        start_method: str = "spawn",
        command_timeout: Optional[float] = None,
    ):
        self.dataset = dataset
        self.num_envs = int(num_envs)
        self.seed_base = int(seed_base)
        self.delay = float(delay)
        self.startup_timeout = float(timeout)
        self.command_timeout = float(
            command_timeout if command_timeout is not None else timeout
        )
        self.start_method = str(start_method)

        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if self.delay < 0:
            raise ValueError("startup delay must be non-negative")
        if self.startup_timeout <= 0 or self.command_timeout <= 0:
            raise ValueError("worker timeouts must be positive")
        if self.start_method != "spawn":
            raise ValueError(
                "Progressive vector rollout requires multiprocessing start_method='spawn'"
            )

        self.ctx = mp.get_context(self.start_method)
        self.processes: List[mp.Process] = []
        self.connections = []
        self.initial_observations = []
        self.vector_steps = 0
        self._closed = False
        self.action_low = None
        self.action_high = None

        try:
            for env_id in range(self.num_envs):
                self._start_one(env_id)
                if env_id + 1 < self.num_envs:
                    time.sleep(self.delay)
            print(
                f"[ENV STARTUP] all {self.num_envs} environments ready",
                flush=True,
            )
        except BaseException:
            self.close(force=True)
            raise

    def _start_one(self, env_id: int) -> None:
        seed = self.seed_base + int(env_id)
        print(
            f"[ENV STARTUP] {env_id + 1:02d}/{self.num_envs} "
            f"STARTING seed={seed}",
            flush=True,
        )

        parent, child = self.ctx.Pipe()

        # A spawn child re-imports Python modules before entering _worker.
        # Therefore set thread limits in the inherited environment *before*
        # process.start(), not only inside _worker.
        previous = {key: os.environ.get(key) for key in _THREAD_ENV}
        os.environ.update(_THREAD_ENV)
        try:
            process = self.ctx.Process(
                target=_worker,
                args=(child, env_id, self.dataset, seed),
                name=f"stage3-env-{env_id:02d}",
            )
            process.daemon = True
            # Append before waiting for READY so timeout / startup failure
            # cleanup cannot leak this just-created process or pipe.
            self.processes.append(process)
            self.connections.append(parent)
            process.start()
        finally:
            child.close()
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        message = self._recv(
            env_id,
            timeout=self.startup_timeout,
            operation=f"startup seed={seed}",
        )
        if message[0] != "READY":
            raise RuntimeError(
                f"env_id={env_id} seed={seed} startup failed: {message}"
            )

        _, obs, low, high = message
        low = np.asarray(low, dtype=np.float32)
        high = np.asarray(high, dtype=np.float32)
        if low.shape != high.shape or not np.all(low < high):
            raise RuntimeError(f"Invalid action bounds from env_id={env_id}")

        if env_id == 0:
            self.action_low = low
            self.action_high = high
        elif not np.array_equal(low, self.action_low) or not np.array_equal(
            high, self.action_high
        ):
            raise RuntimeError("Vector environments have different action bounds")

        self.initial_observations.append(obs)
        print(
            f"[ENV STARTUP] {env_id + 1:02d}/{self.num_envs} READY",
            flush=True,
        )

    def _recv(self, env_id: int, timeout: float, operation: str):
        process = self.processes[env_id]
        conn = self.connections[env_id]

        if not process.is_alive():
            raise RuntimeError(
                f"env_id={env_id} worker died before {operation}; "
                f"exitcode={process.exitcode}"
            )
        if not conn.poll(float(timeout)):
            raise TimeoutError(
                f"env_id={env_id} {operation} exceeded {timeout:.1f}s"
            )
        try:
            message = conn.recv()
        except EOFError as error:
            raise RuntimeError(
                f"env_id={env_id} pipe closed during {operation}; "
                f"exitcode={process.exitcode}"
            ) from error

        if not isinstance(message, tuple) or not message:
            raise RuntimeError(
                f"env_id={env_id} returned malformed message during "
                f"{operation}: {message!r}"
            )
        if message[0] == "ERROR":
            detail = message[3] if len(message) > 3 else repr(message)
            raise RuntimeError(
                f"env_id={env_id} worker ERROR during {operation}\n{detail}"
            )
        return message

    def step(
        self,
        actions: Sequence[np.ndarray],
        env_ids: Optional[Iterable[int]] = None,
    ) -> List[Tuple[int, tuple]]:
        if self._closed:
            raise RuntimeError("Cannot step a closed vector environment")

        ids = (
            list(range(self.num_envs))
            if env_ids is None
            else [int(i) for i in env_ids]
        )
        if len(ids) != len(actions):
            raise ValueError(
                f"env_ids/actions length mismatch: {len(ids)} vs {len(actions)}"
            )
        if len(set(ids)) != len(ids):
            raise ValueError("env_ids contains duplicates")
        if any(i < 0 or i >= self.num_envs for i in ids):
            raise IndexError("env_id out of range")

        for env_id, action in zip(ids, actions):
            action = np.asarray(action, dtype=np.float32)
            if action.shape != self.action_low.shape:
                raise ValueError(
                    f"env_id={env_id} invalid action shape {action.shape}; "
                    f"expected {self.action_low.shape}"
                )
            if not np.isfinite(action).all():
                raise FloatingPointError(
                    f"env_id={env_id} action contains non-finite values"
                )
            self.connections[env_id].send(("step", action))

        results = []
        for env_id in ids:
            message = self._recv(
                env_id,
                timeout=self.command_timeout,
                operation="step",
            )
            if message[0] not in ("OK", "FATAL"):
                raise RuntimeError(
                    f"env_id={env_id} unexpected step response: {message}"
                )
            results.append((env_id, message))

        if ids:
            self.vector_steps += 1
        return results

    def reset(self, env_id: int, seed: int, rebuild: bool = False):
        """Reset one env.  A RESET_FATAL is retried by rebuilding that env."""
        env_id = int(env_id)
        seed = int(seed)
        if env_id < 0 or env_id >= self.num_envs:
            raise IndexError("env_id out of range")

        command = "rebuild" if rebuild else "reset"
        self.connections[env_id].send((command, seed))
        message = self._recv(
            env_id,
            timeout=self.command_timeout,
            operation=f"{command} seed={seed}",
        )

        if message[0] == "RESET_OK":
            return message[1]

        if message[0] == "REBUILD_OK":
            _, obs, low, high = message
            low = np.asarray(low, dtype=np.float32)
            high = np.asarray(high, dtype=np.float32)
            if not np.array_equal(low, self.action_low) or not np.array_equal(
                high, self.action_high
            ):
                raise RuntimeError(
                    f"env_id={env_id} action bounds changed after rebuild"
                )
            return obs

        if message[0] == "RESET_FATAL" and not rebuild:
            # Keep recovery local to the failed env.
            return self.reset(env_id, seed, rebuild=True)

        raise RuntimeError(
            f"env_id={env_id} reset failed with response: {message}"
        )

    def alive_worker_count(self) -> int:
        return sum(int(process.is_alive()) for process in self.processes)

    def close(self, force: bool = False) -> None:
        if self._closed:
            return

        if not force:
            for conn, process in zip(self.connections, self.processes):
                if process.is_alive():
                    try:
                        conn.send(("close", None))
                    except (BrokenPipeError, EOFError, OSError):
                        pass

            for process in self.processes:
                try:
                    process.join(timeout=5.0)
                except BaseException:
                    pass
        else:
            # Startup failure: do not wait 5 seconds per partially created
            # worker before terminating them.
            for process in self.processes:
                if process.is_alive():
                    try:
                        process.terminate()
                    except BaseException:
                        pass

        for process in self.processes:
            if process.is_alive():
                try:
                    process.terminate()
                except BaseException:
                    pass
                try:
                    process.join(timeout=5.0)
                except BaseException:
                    pass

        # Last resort for a worker that ignored terminate().
        for process in self.processes:
            if process.is_alive() and hasattr(process, "kill"):
                try:
                    process.kill()
                    process.join(timeout=2.0)
                except BaseException:
                    pass

        for conn in self.connections:
            try:
                conn.close()
            except BaseException:
                pass

        self._closed = True
