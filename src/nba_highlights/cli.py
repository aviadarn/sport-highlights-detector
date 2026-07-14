import os
from datetime import datetime, timezone

import typer

from nba_highlights.config import Settings
from nba_highlights.eval import GroundTruth, TruthSegment, score_report
from nba_highlights.judge.factory import load_judge
from nba_highlights.judge.labeling import build_judge_inputs, verdicts_to_truth
from nba_highlights.judge.agreement import agreement
from nba_highlights.scoring.factory import load_scorer
from nba_highlights.models import Report
from nba_highlights.pipeline import run_pipeline

app = typer.Typer(help="NBA video highlight detection -> JSON report")


def _build_backends(settings: Settings):
    if settings.asr_backend == "faster-whisper":
        from nba_highlights.transcribe.faster_whisper_client import FasterWhisperTranscriber
        transcriber = FasterWhisperTranscriber(model_size=settings.asr_model)
    else:
        from nba_highlights.transcribe.stub import StubTranscriber
        transcriber = StubTranscriber()
    if settings.action_backend == "videomae":
        from nba_highlights.action.videomae_client import VideoMAEActionRecognizer
        action = VideoMAEActionRecognizer()
    else:
        from nba_highlights.action.stub import StubActionRecognizer
        action = StubActionRecognizer()
    return transcriber, action


@app.command()
def analyze(
    source: str = typer.Argument(..., help="YouTube URL, HLS .m3u8, or file path"),
    out: str = typer.Option("report.json", help="Output report path"),
    workdir: str = typer.Option("./work", help="Scratch dir for media"),
    threshold: float = typer.Option(None, help="Selection threshold override"),
    action_backend: str = typer.Option(None, help="stub | videomae"),
    asr_backend: str = typer.Option(None, help="stub | faster-whisper"),
    full_action: bool = typer.Option(False, help="Run action model on every shot"),
    emit_features: str = typer.Option(None, help="Write per-shot features JSONL to PATH"),
    fusion: str = typer.Option(None, help="weighted | learned"),
    fusion_model: str = typer.Option(None, help="Path to a trained fusion model (.joblib)"),
):
    settings = Settings.from_env()
    if threshold is not None:
        settings.threshold = threshold
    if action_backend is not None:
        settings.action_backend = action_backend
    if asr_backend is not None:
        settings.asr_backend = asr_backend
    settings.full_action = full_action
    if fusion is not None:
        settings.fusion = fusion
    if fusion_model is not None:
        settings.fusion_model = fusion_model
    settings.check()

    transcriber, action = _build_backends(settings)
    scorer = load_scorer(settings.fusion, settings.fusion_model, settings.weights)
    report = run_pipeline(
        source, settings, transcriber, action,
        workdir=workdir,
        generated_at=datetime.now(timezone.utc).isoformat(),
        title="",
        scorer=scorer,
        emit_features_path=emit_features,
    )
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        f.write(report.model_dump_json(indent=2))
    typer.echo(f"wrote {out} ({len(report.highlights)} highlights)")


@app.command()
def eval(
    report: str = typer.Option(..., help="Path to a report.json"),
    truth: str = typer.Option(..., help="Path to a ground-truth JSON"),
    out: str = typer.Option("eval.json", help="Output eval path"),
):
    with open(report) as f:
        rep = Report.model_validate_json(f.read())
    with open(truth) as f:
        gt = GroundTruth.model_validate_json(f.read())
    result = score_report(rep, gt)
    with open(out, "w") as f:
        f.write(result.model_dump_json(indent=2))
    typer.echo(f"precision={result.precision:.3f} recall={result.recall:.3f} "
               f"f1={result.f1:.3f} (tp={result.tp} fp={result.fp} fn={result.fn})")


@app.command()
def label(
    report: str = typer.Option(..., help="Path to a report.json to seed candidates from"),
    out: str = typer.Option(..., help="Output candidate truth JSON"),
    min_score: float = typer.Option(0.4, help="Keep highlights with score >= this"),
):
    with open(report) as f:
        rep = Report.model_validate_json(f.read())
    segs = [TruthSegment(start_s=h.start_s, end_s=h.end_s)
            for h in rep.highlights if h.score >= min_score]
    gt = GroundTruth(highlights=segs)
    with open(out, "w") as f:
        f.write(gt.model_dump_json(indent=2))
    typer.echo(f"wrote {out} ({len(segs)} candidate highlights to prune)")


@app.command()
def judge(
    report: str = typer.Option(..., help="Path to a report.json"),
    video: str = typer.Option(..., help="Path to the local video the report was built from"),
    out: str = typer.Option(..., help="Output GroundTruth truth JSON"),
    workdir: str = typer.Option("./judge-work", help="Scratch dir for judge keyframes"),
    backend: str = typer.Option(None, help="stub | claude"),
    model: str = typer.Option(None, help="Claude model (default claude-opus-4-8)"),
    min_confidence: float = typer.Option(0.6, help="Keep verdicts with confidence >= this"),
    frames_per_clip: int = typer.Option(3, help="Keyframes sampled per candidate"),
):
    settings = Settings.from_env()
    if backend is not None:
        settings.judge_backend = backend
    if model is not None:
        settings.judge_model = model
    settings.check()

    with open(report) as f:
        rep = Report.model_validate_json(f.read())
    judge_backend = load_judge(settings.judge_backend, settings.judge_model)
    inputs = build_judge_inputs(rep, video, workdir, frames_per_clip=frames_per_clip)
    verdicts = [judge_backend.judge(clip) for clip in inputs]
    gt = verdicts_to_truth(rep, verdicts, min_confidence=min_confidence)
    with open(out, "w") as f:
        f.write(gt.model_dump_json(indent=2))
    typer.echo(f"wrote {out} ({len(gt.highlights)}/{len(rep.highlights)} judged highlights)")


@app.command(name="judge-agreement")
def judge_agreement(
    judged: str = typer.Option(..., help="Judge-produced GroundTruth JSON"),
    human: str = typer.Option(..., help="Human-labeled GroundTruth JSON"),
    out: str = typer.Option("agreement.json", help="Output agreement JSON"),
    iou_threshold: float = typer.Option(0.5, help="IoU match threshold"),
):
    with open(judged) as f:
        j = GroundTruth.model_validate_json(f.read())
    with open(human) as f:
        h = GroundTruth.model_validate_json(f.read())
    result = agreement(j, h, iou_threshold=iou_threshold)
    with open(out, "w") as f:
        f.write(result.model_dump_json(indent=2))
    typer.echo(f"precision={result.precision:.3f} recall={result.recall:.3f} "
               f"f1={result.f1:.3f} kappa={result.kappa:.3f} "
               f"(tp={result.tp} fp={result.fp} fn={result.fn})")
