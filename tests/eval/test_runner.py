import json
from pathlib import Path

import pytest
from conftest import GitMinirepo

from reposcout.contracts.localizer import LocalizerConfig
from reposcout.eval import runner
from reposcout.eval.dataset import DatasetRow
from reposcout.eval.report import report
from reposcout.eval.repos import git


def test_runner_real_checkout_error_isolation_and_resume(
    minirepo_git: GitMinirepo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, sha, _ = minirepo_git
    rows = [
        DatasetRow(
            repo="test/postbag",
            instance_id=i,
            base_commit=commit,
            problem_statement="rate_band grams letter_limit_grams",
            patch="--- a/postbag/rates.py\n+++ b/postbag/rates.py\n@@ -10 +10 @@\n-old\n+new\n",
        )
        for i, commit in [("ok", sha), ("bad", "9" * 40)]
    ]
    monkeypatch.setattr(runner, "load_split", lambda split, data_dir: rows)
    cache = tmp_path / "cache"
    mirror = cache / "repos" / "test__postbag.git"
    mirror.parent.mkdir(parents=True)
    git("clone", "--bare", str(root), str(mirror))
    results = tmp_path / "results"
    kwargs = {"results_dir": results, "cache_dir": cache}
    metrics = runner.run("bm25", "dev", "test", **kwargs)
    assert metrics["instances"] == 2 and metrics["error_rate"] == 0.5
    assert metrics["scored_instances"] == 1 and metrics["unavailable_gold"] == 1
    run_dir = results / "test"
    predictions = (run_dir / "predictions.jsonl").read_text().splitlines()
    assert json.loads(predictions[0])["functions"][0] == "postbag/rates.py::RateCard.rate_band"
    assert json.loads((run_dir / "errors.jsonl").read_text())["instance_id"] == "bad"
    assert json.loads((run_dir / "gold.jsonl").read_text())["gold_functions"] == [
        "postbag/rates.py::RateCard.rate_band"
    ]
    monkeypatch.setattr(runner, "_job", lambda *args: pytest.fail("reran cached instance"))
    assert runner.run("bm25", "dev", "test", **kwargs) == metrics
    assert "| test | dev | 2 |" in report(results)
    with pytest.raises(ValueError, match="resume config"):
        runner.run("dense", "dev", "test", **kwargs)


def test_token_guard_estimate_and_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from reposcout.contracts.models import Localization, RunStats

    rows = [
        DatasetRow(
            repo="test/postbag",
            instance_id=str(i),
            base_commit="1" * 40,
            problem_statement="issue",
            patch="",
        )
        for i in range(8)
    ]
    monkeypatch.setattr(runner, "load_split", lambda split, data_dir: rows)

    def job(spec: runner.Job, *args: object) -> tuple:
        issue = spec.row.issue()
        return issue, Localization(
            instance_id=issue.instance_id,
            files=[],
            functions=[],
            candidates=[],
            stats=RunStats(status="complete", prompt_tokens=2),
        )

    monkeypatch.setattr(runner, "_job", job)
    kw = {"cache_dir": tmp_path / "cache", "results_dir": tmp_path / "results", "workers": 1}
    assert runner.run("bm25", "dev", "pilot", max_total_tokens=3, **kw)["instances"] == 2
    assert runner.run("bm25", "dev", "pilot", estimate=True, **kw)["instances"] == 5
    assert runner.run("bm25", "dev", "pilot", estimate=True, yes=True, **kw)["instances"] == 8
    est = json.loads((tmp_path / "results/pilot/estimate.json").read_text())
    assert est["projected_tokens"] == 16


def test_worker_times_out_and_returns_diagnostic(tmp_path: Path) -> None:
    row = DatasetRow(
        instance_id="timeout",
        repo="test/postbag",
        base_commit="1" * 40,
        problem_statement="x",
        patch="",
    )
    job = runner.Job(
        row=row,
        localizer="bm25",
        cache=tmp_path,
        out=tmp_path,
        config=LocalizerConfig(timeout_s=0.000001),
    )
    _, p = runner._job(job, 0, tmp_path, "run")
    assert p.stats.status == "error" and "timeout" in p.stats.error


def test_unknown_localizer_leaves_no_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "load_split", lambda *args: pytest.fail("loaded data for bad name"))
    with pytest.raises(ValueError, match="unknown localizer"):
        runner.run("bmm25", "dev", "typo", results_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_failed_pilot_marks_estimate_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        DatasetRow(
            repo="test/postbag",
            instance_id=str(i),
            base_commit="1" * 40,
            problem_statement="bug",
            patch="",
        )
        for i in range(6)
    ]
    monkeypatch.setattr(runner, "load_split", lambda *args: rows)
    monkeypatch.setattr(
        runner,
        "_job",
        lambda spec, *args: (None, runner.failure(spec.row.instance_id, "unavailable", 0)),
    )
    metrics = runner.run(
        "bm25",
        "dev",
        "bad-pilot",
        estimate=True,
        yes=True,
        cache_dir=tmp_path / "cache",
        results_dir=tmp_path / "results",
    )
    assert metrics["instances"] == 5
    estimate = json.loads((tmp_path / "results/bad-pilot/estimate.json").read_text())
    assert estimate["status"] == "unavailable" and estimate["pilot_errors"] == 5


def test_worker_transport_preserves_agent_configuration(tmp_path: Path) -> None:
    from reposcout.agent.state import AgentConfig
    from reposcout.eval.dataset import DatasetRow

    cfg = AgentConfig(seed=53, explore_limit=3)
    job = runner.Job(
        row=DatasetRow(
            instance_id="one",
            repo="org/repo",
            base_commit="a" * 40,
            problem_statement="issue",
            patch="",
        ),
        localizer="agent",
        cache=tmp_path,
        out=tmp_path,
        config=cfg,
    )
    assert runner.Job.model_validate_json(job.model_dump_json()).config == cfg
