import json
from typer.testing import CliRunner
from nba_highlights.cli import app
from nba_highlights.models import (
    Report, VideoInfo, Highlight, SignalBreakdown, AudioSignals, ActionSignals, Evidence,
)

runner = CliRunner()


def _report(spans_scores):
    hs = [Highlight(id=i, start_s=s, end_s=e, score=sc,
                    signals=SignalBreakdown(audio=AudioSignals(), action=ActionSignals()),
                    evidence=Evidence(), member_scenes=[i])
          for i, (s, e, sc) in enumerate(spans_scores)]
    return Report(video=VideoInfo(source="u"), config={}, generated_at="t", highlights=hs)


def test_label_seeds_truth_from_report(tmp_path):
    rp = tmp_path / "r.json"
    op = tmp_path / "truth.json"
    rp.write_text(_report([(10.0, 20.0, 0.9), (50.0, 55.0, 0.3)]).model_dump_json())
    result = runner.invoke(app, ["label", "--report", str(rp), "--out", str(op),
                                 "--min-score", "0.4"])
    assert result.exit_code == 0, result.output
    truth = json.loads(op.read_text())
    assert truth["highlights"] == [{"start_s": 10.0, "end_s": 20.0}]  # 0.3 dropped


def test_analyze_passes_emit_and_scorer(tmp_path, monkeypatch):
    import nba_highlights.cli as climod

    captured = {}

    def fake_run_pipeline(source, settings, transcriber, action, **kw):
        captured["emit"] = kw.get("emit_features_path")
        captured["scorer"] = kw.get("scorer")
        return Report(video=VideoInfo(source=source), config={}, generated_at="t",
                      highlights=[])

    monkeypatch.setattr(climod, "run_pipeline", fake_run_pipeline)
    out = tmp_path / "r.json"
    result = runner.invoke(app, ["analyze", "/v.mp4", "--out", str(out),
                                 "--workdir", str(tmp_path),
                                 "--emit-features", str(tmp_path / "f.jsonl")])
    assert result.exit_code == 0, result.output
    assert captured["emit"] == str(tmp_path / "f.jsonl")
    assert captured["scorer"] is not None  # a scorer was built and passed
