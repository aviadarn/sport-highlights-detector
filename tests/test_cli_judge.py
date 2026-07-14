import json
from typer.testing import CliRunner
from nba_highlights.cli import app
from nba_highlights.models import (
    Report, VideoInfo, Highlight, SignalBreakdown, AudioSignals, ActionSignals, Evidence,
)

runner = CliRunner()


def _report(spans):
    hs = [Highlight(id=i, start_s=s, end_s=e, score=0.9,
                    signals=SignalBreakdown(audio=AudioSignals(), action=ActionSignals()),
                    evidence=Evidence(transcript=tx), member_scenes=[i])
          for i, (s, e, tx) in enumerate(spans)]
    return Report(video=VideoInfo(source="u"), config={}, generated_at="t", highlights=hs)


def test_judge_command_writes_groundtruth(tmp_path, monkeypatch):
    import nba_highlights.cli as climod
    from nba_highlights.judge.base import JudgeInput

    # avoid real frame extraction: stub build_judge_inputs
    def fake_build(report, video, workdir, **kw):
        return [JudgeInput(scene_id=h.id, start_s=h.start_s, end_s=h.end_s,
                           keyframe_paths=[], transcript=h.evidence.transcript)
                for h in report.highlights]

    monkeypatch.setattr(climod, "build_judge_inputs", fake_build)

    rp = tmp_path / "r.json"
    rp.write_text(_report([(0.0, 6.0, "dunk"), (30.0, 36.0, "nothing")]).model_dump_json())
    out = tmp_path / "truth.json"
    # stub backend: verdicts default False -> mark scene 0 hot via JUDGE env? Instead use
    # the stub's default; here we assert the command runs and writes a valid GroundTruth.
    result = runner.invoke(app, ["judge", "--report", str(rp), "--video", "game.mp4",
                                 "--out", str(out), "--backend", "stub",
                                 "--workdir", str(tmp_path / "jw")])
    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text())
    assert "highlights" in data  # valid GroundTruth shape (stub default -> empty list)


def test_judge_agreement_command(tmp_path):
    from nba_highlights.eval import GroundTruth, TruthSegment
    j = tmp_path / "j.json"
    h = tmp_path / "h.json"
    o = tmp_path / "a.json"
    g = GroundTruth(highlights=[TruthSegment(start_s=0.0, end_s=10.0)])
    j.write_text(g.model_dump_json())
    h.write_text(g.model_dump_json())
    result = runner.invoke(app, ["judge-agreement", "--judged", str(j),
                                 "--human", str(h), "--out", str(o)])
    assert result.exit_code == 0, result.output
    assert json.loads(o.read_text())["f1"] == 1.0
