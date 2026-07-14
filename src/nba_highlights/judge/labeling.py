import os

from nba_highlights import media
from nba_highlights.eval import GroundTruth, TruthSegment
from nba_highlights.judge.base import JudgeInput, JudgeVerdict
from nba_highlights.models import Report


def build_judge_inputs(report: Report, video_path: str, workdir: str,
                       frames_per_clip: int = 3,
                       frame_times_fn=media.frame_times,
                       extract_frame_fn=media.extract_frame) -> list[JudgeInput]:
    os.makedirs(workdir, exist_ok=True)
    inputs: list[JudgeInput] = []
    for h in report.highlights:
        paths: list[str] = []
        for j, t in enumerate(frame_times_fn(h.start_s, h.end_s, frames_per_clip)):
            out = os.path.join(workdir, f"judge_{h.id:04d}_{j:02d}.jpg")
            extract_frame_fn(video_path, t, out)
            paths.append(out)
        inputs.append(JudgeInput(
            scene_id=h.id, start_s=h.start_s, end_s=h.end_s,
            keyframe_paths=paths, transcript=h.evidence.transcript))
    return inputs


def verdicts_to_truth(report: Report, verdicts: list[JudgeVerdict],
                      min_confidence: float = 0.0) -> GroundTruth:
    segs = [TruthSegment(start_s=h.start_s, end_s=h.end_s)
            for h, v in zip(report.highlights, verdicts)
            if v.is_highlight and v.confidence >= min_confidence]
    return GroundTruth(highlights=segs)
