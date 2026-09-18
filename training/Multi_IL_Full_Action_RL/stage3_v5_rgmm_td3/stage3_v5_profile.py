"""Opt-in device-complete timings; disabled timers never synchronize."""
from contextlib import contextmanager
import time
import os
from pathlib import Path


class StageProfiler:
    def __init__(self, device=None, enabled=False):
        self.device, self.enabled = device, bool(enabled)
        self.totals, self.counts = {}, {}

    def synchronize(self):
        if not self.enabled or self.device is None:
            return
        import torch
        if self.device.type == "npu":
            torch.npu.synchronize()
        elif self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def measure(self, name, device=False):
        if not self.enabled:
            yield
            return
        if device:
            self.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            if device:
                self.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            self.totals[name] = self.totals.get(name, 0.0) + elapsed
            self.counts[name] = self.counts.get(name, 0) + 1

    def report(self):
        return {name: {"total_ms": value, "calls": self.counts[name],
                       "mean_ms": value / self.counts[name]}
                for name, value in self.totals.items()}


def cpu_snapshot(processes=()):
    result = {"wall": time.monotonic(), "process_seconds": time.process_time(),
              "available_cores": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()}
    quota = Path("/sys/fs/cgroup/cpu.max")
    if quota.is_file():
        values = quota.read_text().split()
        if values[0] != "max":
            result["cpu_quota_cores"] = int(values[0])/int(values[1])
            result["available_cores"] = min(result["available_cores"], result["cpu_quota_cores"])
    if Path("/proc/stat").is_file():
        values = [int(v) for v in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
        result.update(total=sum(values[:8]), idle=values[3]+values[4])
        ticks = os.sysconf("SC_CLK_TCK")
        for process in processes:
            try:
                fields = Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()
                result["process_seconds"] += (int(fields[11])+int(fields[12])) / ticks
            except (OSError, IndexError, ValueError):
                pass
    return result


def cpu_measurement(before, after):
    wall = max(after["wall"] - before["wall"], 1e-9)
    cpu = max(after["process_seconds"] - before["process_seconds"], 0.0)
    result = {"process_cpu_core_percent": cpu / wall * 100,
              "available_cores": after["available_cores"],
              "available_core_utilization_percent": cpu / wall / max(1, after["available_cores"]) * 100}
    if "total" in after:
        result["host_cpu_utilization_percent"] = 100 * (1 - (after["idle"] - before["idle"]) /
            max(1, after["total"] - before["total"]))
    return result
