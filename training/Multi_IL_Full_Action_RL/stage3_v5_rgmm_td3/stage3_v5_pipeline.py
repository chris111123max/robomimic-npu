"""Transition-credit backpressure and short, polled learner bursts."""
import math
import time


class TransitionCredit:
    def __init__(self, utd=0.25, limit=256, pending=0.0):
        self.utd, self.limit, self.pending = float(utd), int(limit), float(pending)
        self.collector_transition_head = self.pending / self.utd
        self.learner_consumed_transition_equivalent = 0.0
        self.max_collector_lag_seen = self.lag
        self.collector_throttle_count = 0

    @property
    def lag(self):
        return max(0.0, self.pending / self.utd)

    @property
    def updates_due(self):
        return int(math.floor(self.pending + 1e-9))

    def collect(self, transitions):
        self.collector_transition_head += int(transitions)
        self.pending += int(transitions) * self.utd
        self.max_collector_lag_seen = max(self.max_collector_lag_seen, self.lag)
        if self.lag > self.limit + 1e-8:
            raise RuntimeError("Collector exceeded transition-credit lag bound")

    def consume(self):
        if not self.updates_due:
            raise RuntimeError("Learner attempted to consume nonexistent credit")
        self.pending -= 1.0
        self.learner_consumed_transition_equivalent += 1.0 / self.utd

    def must_throttle(self, next_count):
        return self.lag >= self.limit or self.lag + int(next_count) > self.limit

    def metrics(self):
        return {"collector_transition_head": self.collector_transition_head,
                "learner_consumed_transition_equivalent": self.learner_consumed_transition_equivalent,
                "collector_lag_transitions": self.lag,
                "max_collector_lag_seen": self.max_collector_lag_seen,
                "collector_throttle_count": self.collector_throttle_count}


def overlap_burst(vector, credit, learn_once, max_updates, intervals=None):
    """One atomic update then poll, never drain an unlimited backlog in flight.

    The first update is allowed even when a worker has just completed; this
    bounds collector delay to one optimizer update rather than starving the
    learner when simulation is faster. Subsequent updates require no ready
    worker. At most one previous round's credits are consumed.
    """
    used = 0
    while used < int(max_updates) and credit.updates_due:
        if used and vector.any_ready():
            break
        start = time.perf_counter()
        if not learn_once():
            break
        end = time.perf_counter()
        if intervals is not None:
            measured = getattr(learn_once, "last_device_interval", (start, end))
            if measured is not None:
                intervals.append(measured)
        credit.consume()
        used += 1
    return used


def interval_overlap(learner_intervals, results):
    """Measured device-complete learner intervals versus worker simulation."""
    workers = []
    for _, message in results:
        if message[0] == "OK":
            info = message[-1]
            if isinstance(info, dict) and "_stage3_env_started" in info:
                workers.append((info["_stage3_env_started"], info["_stage3_env_finished"]))
    overlaps = [max(0.0, min(b, d) - max(a, c))
                for a, b in learner_intervals for c, d in workers]
    return max(overlaps, default=0.0) * 1000
