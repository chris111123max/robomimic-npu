from helpers import diagnostic_snapshot
import os,time,traceback
from typing import Optional
from multiprocessing.shared_memory import SharedMemory
import numpy as np
from stage3_v5_vector_env import _THREAD_ENV,_worker_cpu_affinity,_safe_send,_action_bounds,_obs_to_flat_shared
def snapshot_worker(conn, env_id: int, dataset: str, initial_seed: int,
            shared_obs_name: Optional[str] = None,
            shared_action_name: Optional[str] = None,
            shared_num_envs: int = 0,
            shared_obs_dim: int = 59,
            shared_action_dim: int = 14) -> None:
    """Own exactly one CPU MuJoCo environment."""
    # These values are also injected by the parent before spawn so that they
    # are already visible while the child imports numpy / MuJoCo.
    os.environ.update(_THREAD_ENV)
    _worker_cpu_affinity(env_id)

    env = None
    obs_shm = action_shm = None
    shared_obs = shared_action = None
    try:
        if shared_obs_name and shared_action_name:
            obs_shm = SharedMemory(name=shared_obs_name)
            action_shm = SharedMemory(name=shared_action_name)
            shared_obs = np.ndarray((int(shared_num_envs), int(shared_obs_dim)),
                                    dtype=np.float32, buffer=obs_shm.buf)
            shared_action = np.ndarray((int(shared_num_envs), int(shared_action_dim)),
                                       dtype=np.float32, buffer=action_shm.buf)
        from stage3_new_evaluation import (
            build_env,
            close_env,
            mujoco_fatal_error_type,
            reset_seed,
            success,
        )

        env = build_env(dataset)
        obs = reset_seed(env, int(initial_seed))
        if shared_obs is not None:
            shared_obs[env_id, :] = _obs_to_flat_shared(obs)
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
                    started = time.perf_counter()
                    obs, reward, done, info = env.step(payload)
                    info = dict(info or {}, _stage3_env_started=started,
                                _stage3_env_finished=time.perf_counter())
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

            elif command == "step_shared":
                try:
                    started = time.perf_counter()
                    obs, reward, done, info = env.step(
                        np.asarray(shared_action[env_id], dtype=np.float32))
                    info = dict(info or {}, _stage3_env_started=started,
                                _stage3_env_finished=time.perf_counter())
                    shared_obs[env_id, :] = _obs_to_flat_shared(obs)
                    _safe_send(conn, ("OK_SHARED", float(reward), bool(done),
                                      bool(success(env)), info))
                except mujoco_fatal_error_type() as error:
                    _safe_send(conn, ("FATAL", str(error)))

            elif command == "reset":
                try:
                    obs = reset_seed(env, int(payload))
                    if shared_obs is not None:
                        shared_obs[env_id, :] = _obs_to_flat_shared(obs)
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
                    if shared_obs is not None:
                        shared_obs[env_id, :] = _obs_to_flat_shared(obs)
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

            elif command == "snapshot":
                _safe_send(conn, ("SNAPSHOT", diagnostic_snapshot(env)))

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
        for resource in (obs_shm, action_shm):
            if resource is not None:
                try:
                    resource.close()
                except BaseException:
                    pass

