from pydantic import BaseModel
from nba_highlights.models import Report


class TruthSegment(BaseModel):
    start_s: float
    end_s: float


class GroundTruth(BaseModel):
    highlights: list[TruthSegment]


class EvalResult(BaseModel):
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    if inter <= 0.0:
        return 0.0
    union = (a_end - a_start) + (b_end - b_start) - inter
    return inter / union if union > 0 else 0.0


def score_report(report: Report, gt: GroundTruth,
                 iou_threshold: float = 0.5) -> EvalResult:
    used: set[int] = set()
    tp = 0
    for h in report.highlights:
        best_i, best_iou = -1, 0.0
        for i, t in enumerate(gt.highlights):
            if i in used:
                continue
            v = iou(h.start_s, h.end_s, t.start_s, t.end_s)
            if v >= iou_threshold and v > best_iou:
                best_i, best_iou = i, v
        if best_i >= 0:
            used.add(best_i)
            tp += 1
    fp = len(report.highlights) - tp
    fn = len(gt.highlights) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return EvalResult(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn)
