from nba_highlights.interfaces import ActionResult

BASKETBALL_CLASSES = {
    "dunking basketball",
    "shooting basketball",
    "playing basketball",
    "dribbling basketball",
}


def score_from_probs(label_probs: dict[str, float]) -> ActionResult:
    basket = {k: v for k, v in label_probs.items() if k in BASKETBALL_CLASSES}
    if basket:
        label = max(basket, key=basket.get)
        score = float(basket[label])
    else:
        label, score = "", 0.0
    top_k = sorted(label_probs.items(), key=lambda kv: kv[1], reverse=True)[:3]
    return ActionResult(score=score, label=label,
                        top_k=[(k, float(v)) for k, v in top_k])
