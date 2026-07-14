from pydantic import BaseModel
from nba_highlights.eval import GroundTruth, iou


class AgreementResult(BaseModel):
    precision: float
    recall: float
    f1: float
    kappa: float
    tp: int
    fp: int
    fn: int
    n_judged: int
    n_human: int


def agreement(judged: GroundTruth, human: GroundTruth,
              iou_threshold: float = 0.5) -> AgreementResult:
    used: set[int] = set()
    tp = 0
    for j in judged.highlights:
        best_i, best_iou = -1, 0.0
        for i, h in enumerate(human.highlights):
            if i in used:
                continue
            v = iou(j.start_s, j.end_s, h.start_s, h.end_s)
            if v >= iou_threshold and v > best_iou:
                best_i, best_iou = i, v
        if best_i >= 0:
            used.add(best_i)
            tp += 1
    fp = len(judged.highlights) - tp
    fn = len(human.highlights) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    n = tp + fp + fn
    if n == 0:
        kappa = 0.0
    else:
        po = tp / n
        pe = ((tp + fp) / n) * ((tp + fn) / n)
        kappa = (po - pe) / (1 - pe) if pe != 1 else 0.0

    return AgreementResult(precision=precision, recall=recall, f1=f1, kappa=kappa,
                           tp=tp, fp=fp, fn=fn,
                           n_judged=len(judged.highlights), n_human=len(human.highlights))
