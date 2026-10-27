#!/usr/bin/env python3
"""Work-queue dispatcher for the PhysX train cohort across Brev hosts.

Each case runs as its own one-case shard of the existing supervisor
(``--shard-count N --shard-index i``), so receipts, validation and the final
merge stay exactly the agent's code. This file only decides *where and when*
each case runs:

* hosts pull the next case when a slot frees up (no static modulo shards, so
  the slowest shard no longer sets the end time);
* a failed case is retried on whichever host is free next, at most
  ``max_retries`` times, into a fresh output root;
* each passing case root is copied back as soon as it finishes;
* with ``--stop-idle``, a host is ``brev stop``-ed once the queue is empty and
  its last case is copied back (never deleted);
* the run state is written to ``state.json`` after every change, so the
  dispatcher can be restarted without relaunching finished cases.

Subcommands: ``plan`` (cost/wall table), ``run`` (dispatch), ``merge-cmd``
(print the agent's merge command for the passing roots).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Protocol


# ---------------------------------------------------------------- config


@dataclass
class Host:
    name: str
    price_per_hour: float
    slots: int = 1  # >1 needs the supervisor's other-Isaac-process guard relaxed


@dataclass
class Config:
    run_tag: str
    case_count: int
    hosts: list[Host]
    remote_runs_dir: str
    launch_template: str  # placeholders: {shard_count} {shard_index} {output_root} {lock}
    local_results_dir: str
    max_retries: int = 1
    poll_seconds: float = 60.0
    shader_cache_tar: str | None = None  # local tar of a warm Kit cache
    shader_cache_remote_dir: str | None = None
    merge_command_prefix: list[str] = field(default_factory=list)

    @staticmethod
    def load(path: Path) -> "Config":
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["hosts"] = [Host(**h) for h in raw["hosts"]]
        config = Config(**raw)
        if not 2 <= config.case_count <= 18:
            raise ValueError("case_count must be in [2, 18] (supervisor shard bounds)")
        for key in ("{shard_count}", "{shard_index}", "{output_root}", "{lock}"):
            if key not in config.launch_template:
                raise ValueError(f"launch_template lacks {key}")
        return config


# ---------------------------------------------------------------- executors


class Executor(Protocol):
    def run(self, host: str, command: str) -> tuple[int, str]: ...
    def copy_from(self, host: str, remote: str, local: str) -> int: ...
    def copy_to(self, host: str, local: str, remote: str) -> int: ...
    def stop(self, host: str) -> int: ...


class BrevExecutor:
    """Thin wrapper over the brev CLI with retry on API rate limits."""

    def __init__(self, attempts: int = 4, backoff_s: float = 15.0):
        self.attempts = attempts
        self.backoff_s = backoff_s

    def _brev(self, *args: str) -> tuple[int, str]:
        out = ""
        for attempt in range(self.attempts):
            proc = subprocess.run(
                ["brev", *args], capture_output=True, text=True, timeout=900
            )
            out = proc.stdout + proc.stderr
            if "rate limit" not in out.lower():
                return proc.returncode, out
            time.sleep(self.backoff_s * (attempt + 1))
        return 1, out

    def run(self, host: str, command: str) -> tuple[int, str]:
        return self._brev("exec", host, command)

    def copy_from(self, host: str, remote: str, local: str) -> int:
        return self._brev("copy", f"{host}:{remote}", local)[0]

    def copy_to(self, host: str, local: str, remote: str) -> int:
        return self._brev("copy", local, f"{host}:{remote}")[0]

    def stop(self, host: str) -> int:
        return self._brev("stop", host)[0]


class FakeExecutor:
    """Simulated hosts for dry runs and tests; time comes from a FakeClock."""

    def __init__(self, clock: "FakeClock", case_minutes: float = 15.0,
                 jitter: float = 0.15, fail_rate: float = 0.0, seed: int = 0):
        self.clock = clock
        self.case_s = case_minutes * 60
        self.jitter = jitter
        self.fail_rate = fail_rate
        self.rng = random.Random(seed)
        self.jobs: dict[str, tuple[float, int]] = {}  # done-marker -> (finish_t, rc)
        self.stopped: list[str] = []
        self.log: list[tuple[float, str, str]] = []

    def run(self, host: str, command: str) -> tuple[int, str]:
        self.log.append((self.clock.now(), host, command))
        if command.startswith("LAUNCH "):
            marker = command.split()[1]
            duration = self.case_s * (1 + self.rng.uniform(-self.jitter, self.jitter))
            rc = 1 if self.rng.random() < self.fail_rate else 0
            self.jobs[marker] = (self.clock.now() + duration, rc)
            return 0, "launched"
        if command.startswith("POLL "):
            marker = command.split()[1]
            finish_t, rc = self.jobs[marker]
            return (0, f"DONE {rc}") if self.clock.now() >= finish_t else (0, "RUNNING")
        return 0, ""

    def copy_from(self, host: str, remote: str, local: str) -> int:
        return 0

    def copy_to(self, host: str, local: str, remote: str) -> int:
        return 0

    def stop(self, host: str) -> int:
        self.stopped.append(host)
        return 0


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


# ---------------------------------------------------------------- remote commands


def output_root(config: Config, index: int, attempt: int) -> str:
    suffix = "" if attempt == 0 else f"-r{attempt}"
    return (f"{config.remote_runs_dir}/{config.run_tag}-n{config.case_count}"
            f"-i{index:02d}{suffix}")


def launch_command(config: Config, index: int, attempt: int, slot: int) -> str:
    root = output_root(config, index, attempt)
    supervisor = config.launch_template.format(
        shard_count=config.case_count,
        shard_index=index,
        output_root=shlex.quote(root),
        lock=shlex.quote(f"/tmp/{config.run_tag}-slot{slot}.lock"),
    )
    # The done marker sits next to the root: the supervisor requires the root
    # itself to be absent at launch.
    done = shlex.quote(root + ".done")
    log = shlex.quote(root + ".launch.log")
    body = f"{supervisor} > {log} 2>&1; echo $? > {done}"
    return f"nohup bash -c {shlex.quote(body)} < /dev/null > /dev/null 2>&1 & echo PID=$!"


def poll_command(config: Config, index: int, attempt: int) -> str:
    done = shlex.quote(output_root(config, index, attempt) + ".done")
    return f"if test -f {done}; then echo DONE $(cat {done}); else echo RUNNING; fi"


# ---------------------------------------------------------------- scheduler


@dataclass
class Job:
    index: int
    attempt: int = 0
    status: str = "queued"  # queued | running | passed | failed
    host: str | None = None
    slot: int | None = None
    started_at: float | None = None
    finished_at: float | None = None
    history: list[dict] = field(default_factory=list)


class Dispatcher:
    def __init__(self, config: Config, executor: Executor, state_path: Path,
                 now: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep,
                 stop_idle: bool = False, fake: bool = False):
        self.config = config
        self.ex = executor
        self.state_path = state_path
        self.now = now
        self.sleep = sleep
        self.stop_idle = stop_idle
        self.fake = fake
        self.jobs = [Job(i) for i in range(config.case_count)]
        self.prewarmed: set[str] = set()
        self.stopped: set[str] = set()
        self.host_started: dict[str, float] = {}
        if state_path.exists():
            self._load()

    # -- persistence
    def _save(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "jobs": [asdict(j) for j in self.jobs],
            "prewarmed": sorted(self.prewarmed),
            "stopped": sorted(self.stopped),
            "host_started": self.host_started,
        }, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    def _load(self) -> None:
        raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.jobs = [Job(**j) for j in raw["jobs"]]
        self.prewarmed = set(raw["prewarmed"])
        self.stopped = set(raw["stopped"])
        self.host_started = raw.get("host_started", {})

    # -- helpers
    def _busy(self, host: str) -> list[Job]:
        return [j for j in self.jobs if j.status == "running" and j.host == host]

    def _free_slots(self, host: Host) -> list[int]:
        used = {j.slot for j in self._busy(host.name)}
        return [s for s in range(host.slots) if s not in used]

    def _prewarm(self, host: str) -> None:
        if host in self.prewarmed or not self.config.shader_cache_tar:
            self.prewarmed.add(host)
            return
        remote_tar = "/tmp/kit-shader-cache.tar"
        if self.ex.copy_to(host, self.config.shader_cache_tar, remote_tar) == 0:
            dest = shlex.quote(self.config.shader_cache_remote_dir or "")
            self.ex.run(host, f"mkdir -p {dest} && tar -xf {remote_tar} -C {dest}")
        self.prewarmed.add(host)

    def _launch(self, job: Job, host: Host, slot: int) -> None:
        self._prewarm(host.name)
        if self.fake:
            cmd = f"LAUNCH {job.index}.{job.attempt}"
        else:
            cmd = launch_command(self.config, job.index, job.attempt, slot)
        rc, out = self.ex.run(host.name, cmd)
        if rc != 0:
            job.history.append({"event": "launch_error", "host": host.name, "out": out[-500:]})
            return  # stays queued; next loop retries the launch
        job.status, job.host, job.slot, job.started_at = "running", host.name, slot, self.now()
        self.host_started.setdefault(host.name, self.now())

    def _poll(self, job: Job) -> None:
        if self.fake:
            cmd = f"POLL {job.index}.{job.attempt}"
        else:
            cmd = poll_command(self.config, job.index, job.attempt)
        rc, out = self.ex.run(job.host, cmd)
        if rc != 0 or "DONE" not in out:
            return
        exit_code = int(out.split("DONE", 1)[1].split()[0])
        record = {"attempt": job.attempt, "host": job.host, "exit": exit_code,
                  "started_at": job.started_at, "finished_at": self.now()}
        if exit_code == 0:
            root = output_root(self.config, job.index, job.attempt)
            local = Path(self.config.local_results_dir)
            if not self.fake:
                local.mkdir(parents=True, exist_ok=True)
            if self.ex.copy_from(job.host, root, str(local) + "/") != 0:
                record["copy"] = "failed; will retry copy on next poll"
                job.history.append(record)
                return
            job.status, job.finished_at = "passed", self.now()
        elif job.attempt < self.config.max_retries:
            job.attempt += 1
            job.status, job.host, job.slot, job.started_at = "queued", None, None, None
        else:
            job.status, job.finished_at = "failed", self.now()
        job.history.append(record)

    def _maybe_stop(self) -> None:
        if not self.stop_idle or any(j.status == "queued" for j in self.jobs):
            return
        for host in self.config.hosts:
            if host.name not in self.stopped and not self._busy(host.name):
                if self.ex.stop(host.name) == 0:
                    self.stopped.add(host.name)
                    self.host_started[host.name + ":stopped"] = self.now()

    # -- main loop
    def step(self) -> bool:
        """One scheduling pass; returns True while work remains."""
        for job in [j for j in self.jobs if j.status == "running"]:
            self._poll(job)
        queue = [j for j in self.jobs if j.status == "queued"]
        for host in self.config.hosts:
            if host.name in self.stopped:
                continue
            for slot in self._free_slots(host):
                if not queue:
                    break
                self._launch(queue.pop(0), host, slot)
        self._maybe_stop()
        self._save()
        return any(j.status in ("queued", "running") for j in self.jobs)

    def run(self) -> dict:
        while self.step():
            self.sleep(self.config.poll_seconds)
        self._maybe_stop()
        self._save()
        return self.summary()

    def summary(self) -> dict:
        passed = [j for j in self.jobs if j.status == "passed"]
        failed = [j for j in self.jobs if j.status == "failed"]
        ends = [j.finished_at for j in self.jobs if j.finished_at is not None]
        starts = [j.started_at or 0 for j in self.jobs if j.history]
        return {
            "passed": len(passed),
            "failed": [j.index for j in failed],
            "retries": sum(j.attempt for j in self.jobs),
            "wall_minutes": round((max(ends) - min(starts)) / 60, 1) if ends else None,
            "passing_roots": [output_root(self.config, j.index, j.attempt) for j in passed],
        }


# ---------------------------------------------------------------- planning


def plan_table(case_count: int, case_minutes: float, overhead_minutes: float,
               offers: list[tuple[str, float]], host_counts: list[int]) -> list[dict]:
    """Wall and cost for k one-slot hosts pulling from one queue.

    Assumes per-minute billing, identical case times and no failures, so it is
    a lower bound; add one case time per expected retry.
    """
    rows = []
    for name, price in offers:
        for k in host_counts:
            if k > case_count:
                continue
            wall = overhead_minutes + math.ceil(case_count / k) * case_minutes
            rows.append({"offer": name, "hosts": k, "wall_min": round(wall),
                         "cost_usd": round(k * price * wall / 60, 2)})
    return rows


# ---------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="wall/cost table for host counts and offers")
    p.add_argument("--cases", type=int, default=18)
    p.add_argument("--case-minutes", type=float, default=15.0)
    p.add_argument("--overhead-minutes", type=float, default=15.0,
                   help="boot + payload copy + cold shader start per host")
    p.add_argument("--offer", action="append", default=[],
                   help="name=price_per_hour; repeatable")
    p.add_argument("--hosts", default="1,2,3,6,9,18")

    r = sub.add_parser("run", help="dispatch the cohort")
    r.add_argument("config", type=Path)
    r.add_argument("--state", type=Path)
    r.add_argument("--stop-idle", action="store_true",
                   help="brev stop each host once the queue is drained (never deletes)")
    r.add_argument("--dry-run", action="store_true",
                   help="simulate with FakeExecutor; touches no host")
    r.add_argument("--case-minutes", type=float, default=15.0, help="dry-run only")
    r.add_argument("--fail-rate", type=float, default=0.0, help="dry-run only")

    m = sub.add_parser("merge-cmd", help="print the agent's merge command")
    m.add_argument("config", type=Path)
    m.add_argument("--state", type=Path)

    args = parser.parse_args(argv)

    if args.cmd == "plan":
        offers = [(o.split("=")[0], float(o.split("=")[1])) for o in args.offer] or [
            ("aws g6e.xlarge", 2.23)]
        rows = plan_table(args.cases, args.case_minutes, args.overhead_minutes,
                          offers, [int(x) for x in args.hosts.split(",")])
        print(f"{'offer':28} {'hosts':>5} {'wall_min':>8} {'cost_usd':>8}")
        for row in rows:
            print(f"{row['offer']:28} {row['hosts']:>5} {row['wall_min']:>8} {row['cost_usd']:>8}")
        return 0

    config = Config.load(args.config)
    state = args.state or args.config.with_name(f"{config.run_tag}.state.json")

    if args.cmd == "merge-cmd":
        d = Dispatcher(config, BrevExecutor(), state)
        roots = d.summary()["passing_roots"]
        if len(roots) != config.case_count:
            print(f"only {len(roots)}/{config.case_count} cases passed", file=sys.stderr)
            return 1
        local = [str(Path(config.local_results_dir) / Path(r).name) for r in roots]
        cmd = list(config.merge_command_prefix)
        for root in local:
            cmd += ["--shard-root", root]
        cmd += ["--expected-shard-count", str(config.case_count)]
        print(shlex.join(cmd))
        return 0

    if args.dry_run:
        clock = FakeClock()
        ex = FakeExecutor(clock, case_minutes=args.case_minutes, fail_rate=args.fail_rate)
        state = state.with_suffix(".dryrun.json")
        state.unlink(missing_ok=True)
        d = Dispatcher(config, ex, state, now=clock.now, sleep=clock.sleep,
                       stop_idle=args.stop_idle, fake=True)
    else:
        d = Dispatcher(config, BrevExecutor(), state, stop_idle=args.stop_idle)
    print(json.dumps(d.run(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
