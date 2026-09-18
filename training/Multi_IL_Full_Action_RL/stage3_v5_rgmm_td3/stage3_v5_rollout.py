"""Parameter snapshots stay fixed for each recurrent reset block."""
import copy
from collections import defaultdict


class BoundarySnapshotExecutor:
    def __init__(self, train_actor, scale, offset, num_envs, horizon=10,
                 max_policy_lag=32, executor_factory=None):
        if executor_factory is None:
            from stage3_v5_actor import BatchedGMMExecutor
            executor_factory = BatchedGMMExecutor
        self.train_actor, self.factory = train_actor, executor_factory
        self.scale, self.offset = scale, offset
        self.num_envs, self.horizon = int(num_envs), int(horizon)
        self.max_policy_lag = int(max_policy_lag)
        self.counters = [0] * self.num_envs
        self.versions = [None] * self.num_envs
        self.executors = {}
        self.train_policy_version = 0
        self.max_policy_version_lag = 0

    def reset_indices(self, indices):
        for i in indices:
            i = int(i)
            if self.versions[i] in self.executors:
                self.executors[self.versions[i]].reset_indices([i])
            self.counters[i], self.versions[i] = 0, None

    def actions_for(self, indices, observations, *args, train_policy_version=None):
        if train_policy_version is not None:
            self.train_policy_version = int(train_policy_version)
        version = self.train_policy_version
        groups = defaultdict(list)
        for position, env_id in enumerate(indices):
            if self.counters[env_id] % self.horizon == 0:
                if version not in self.executors:
                    snapshot = copy.deepcopy(self.train_actor).eval()
                    snapshot.requires_grad_(False)
                    self.executors[version] = self.factory(
                        snapshot, self.scale, self.offset, self.num_envs, self.horizon)
                old = self.versions[env_id]
                if old in self.executors:
                    self.executors[old].reset_indices([env_id])
                self.versions[env_id] = version
                self.executors[version].reset_indices([env_id])
            lag = version - self.versions[env_id]
            self.max_policy_version_lag = max(self.max_policy_version_lag, lag)
            if lag > self.max_policy_lag:
                raise RuntimeError("Rollout policy lag exceeded block/credit bound")
            groups[self.versions[env_id]].append((position, env_id))
        actions = [None] * len(indices)
        for block_version, rows in groups.items():
            values = self.executors[block_version].actions_for(
                [i for _, i in rows], [observations[p] for p, _ in rows], *args)
            for (position, env_id), action in zip(rows, values):
                actions[position] = action
                self.counters[env_id] += 1
        active = set(self.versions)
        for old in list(self.executors):
            if old not in active:
                del self.executors[old]
        return actions

    def metrics(self):
        versions = [v for v in self.versions if v is not None]
        oldest = min(versions, default=self.train_policy_version)
        return {"train_policy_version": self.train_policy_version,
                "rollout_policy_version": oldest,
                "rollout_policy_versions": list(self.versions),
                "policy_version_lag": self.train_policy_version - oldest,
                "max_policy_version_lag": self.max_policy_version_lag}
