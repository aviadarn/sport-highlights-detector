import json
from typer.testing import CliRunner
from nba_highlights.cli import app
from nba_highlights.models import (
    Report, VideoInfo, Highlight, SignalBreakdown, AudioSignals, ActionSignals, Evidence,
)

runner = CliRunner()


def test_analyze_stub_end_to_end(tmp_path, monkeypatch):
    # stub backends need no models; force a trivial file source through a fake ingest
    import nba_highlights.cli as climod

    def fake_run_pipeline(source, settings, transcriber, action, **kw):
        assert settings.action_backend == "stub"
        return Report(video=VideoInfo(source=source), config={}, generated_at="t",
                      highlights=[])

    monkeypatch.setattr(climod, "run_pipeline", fake_run_pipeline)
    out = tmp_path / "r.json"
    result = runner.invoke(app, ["analyze", "/v.mp4", "--out", str(out),
                                 "--workdir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text())["video"]["source"] == "/v.mp4"


def test_eval_command(tmp_path):
    report = Report(
        video=VideoInfo(source="u"), config={}, generated_at="t",
        highlights=[Highlight(id=0, start_s=0.0, end_s=10.0, score=0.9,
                    signals=SignalBreakdown(audio=AudioSignals(), action=ActionSignals()),
                    evidence=Evidence(), member_scenes=[0])])
    truth = {"highlights": [{"start_s": 0.0, "end_s": 10.0}]}
    rp = tmp_path / "report.json"
    tp = tmp_path / "truth.json"
    ep = tmp_path / "eval.json"
    rp.write_text(report.model_dump_json())
    tp.write_text(json.dumps(truth))
    result = runner.invoke(app, ["eval", "--report", str(rp), "--truth", str(tp),
                                 "--out", str(ep)])
    assert result.exit_code == 0, result.output
    assert json.loads(ep.read_text())["f1"] == 1.0
