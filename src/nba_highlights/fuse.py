from nba_highlights.config import Weights


def percentile_normalize(values: list[float]) -> list[float]:
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [0.0]
    lo = min(values)
    hi = max(values)
    if hi - lo == 0:
        return [0.0] * n
    # Ties are broken by original index (stable sort): equal inputs get distinct ranks, deterministically.
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    for rank, i in enumerate(order):
        ranks[i] = rank / (n - 1)
    return ranks


def audio_composite(loudness_n: float, prosody_n: float, keywords_n: float,
                    w: Weights) -> float:
    return w.wL * loudness_n + w.wP * prosody_n + w.wK * keywords_n


def fuse(audio_comp: float, action: float, w: Weights) -> float:
    return w.wA * audio_comp + w.wV * action
