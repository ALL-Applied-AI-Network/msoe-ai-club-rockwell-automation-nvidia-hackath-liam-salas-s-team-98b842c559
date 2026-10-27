"""Scheduler checks with simulated hosts; no Brev or Isaac involved."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import dispatch as d  # noqa: E402

EXAMPLE = Path(__file__).with_name("hosts.example.json")


def _config(tmp_path, hosts=3, cases=18, retries=1):
    raw = json.loads(EXAMPLE.read_text())
    raw["case_count"] = cases
    raw["max_retries"] = retries
    raw["hosts"] = [{"name": f"h{i}", "price_per_hour": 2.23} for i in range(hosts)]
    raw["local_results_dir"] = str(tmp_path / "results")
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(raw))
    return d.Config.load(path)


def _dispatcher(tmp_path, config, fail_rate=0.0, seed=0, stop_idle=True):
    clock = d.FakeClock()
    ex = d.FakeExecutor(clock, case_minutes=15, fail_rate=fail_rate, seed=seed)
    disp = d.Dispatcher(config, ex, tmp_path / "state.json", now=clock.now,
                        sleep=clock.sleep, stop_idle=stop_idle, fake=True)
    return disp, ex


def test_all_cases_pass_once_each(tmp_path):
    disp, ex = _dispatcher(tmp_path, _config(tmp_path))
    summary = disp.run()
    assert summary["passed"] == 18 and summary["failed"] == []
    launches = [c for _, _, c in ex.log if c.startswith("LAUNCH")]
    assert len(launches) == 18 and len(set(launches)) == 18


def test_never_two_jobs_in_one_slot(tmp_path):
    disp, _ = _dispatcher(tmp_path, _config(tmp_path))
    while disp.step():
        busy = [(j.host, j.slot) for j in disp.jobs if j.status == "running"]
        assert len(busy) == len(set(busy))
        disp.sleep(60)


def test_failure_retried_into_fresh_root_then_capped(tmp_path):
    config = _config(tmp_path, retries=1)
    disp, _ = _dispatcher(tmp_path, config, fail_rate=1.0)
    summary = disp.run()
    assert summary["passed"] == 0 and len(summary["failed"]) == 18
    assert all(j.attempt == 1 for j in disp.jobs)
    assert d.output_root(config, 3, 1).endswith("-i03-r1")


def test_hosts_stopped_only_after_queue_drains(tmp_path):
    disp, ex = _dispatcher(tmp_path, _config(tmp_path, hosts=4, cases=6))
    disp.run()
    assert sorted(ex.stopped) == ["h0", "h1", "h2", "h3"]
    # h2/h3 get one case each and stop while h0/h1 still run their second.
    first_stop = min(v for k, v in disp.host_started.items() if k.endswith(":stopped"))
    last_finish = max(j.finished_at for j in disp.jobs)
    assert first_stop < last_finish


def test_state_resume_does_not_relaunch_passed(tmp_path):
    config = _config(tmp_path, cases=4)
    disp, _ = _dispatcher(tmp_path, config)
    disp.run()
    again, ex2 = _dispatcher(tmp_path, config)
    again.run()
    assert not [c for _, _, c in ex2.log if c.startswith("LAUNCH")]


def test_launch_command_keeps_root_absent_and_quotes(tmp_path):
    config = _config(tmp_path)
    cmd = d.launch_command(config, 5, 0, 0)
    assert "--shard-count 18" in cmd and "--shard-index 5" in cmd
    assert "-i05.done" in cmd and cmd.startswith("nohup bash -c ")


def test_plan_table_queue_math():
    rows = d.plan_table(18, 15, 15, [("x", 2.0)], [3, 18])
    assert rows[0] == {"offer": "x", "hosts": 3, "wall_min": 105, "cost_usd": 10.5}
    assert rows[1]["wall_min"] == 30
