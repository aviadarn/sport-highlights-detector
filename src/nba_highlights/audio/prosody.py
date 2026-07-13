import numpy as np


def _zero_crossing_rate(frame: np.ndarray) -> float:
    if len(frame) < 2:
        return 0.0
    signs = np.sign(frame)
    signs[signs == 0] = 1
    return float(np.mean(np.abs(np.diff(signs))) / 2.0)


def shot_prosody(samples: np.ndarray, sr: int,
                 start_s: float, end_s: float) -> float:
    a = max(0, int(start_s * sr))
    b = min(len(samples), int(end_s * sr))
    frame = samples[a:b]
    if len(frame) == 0:
        return 0.0
    energy = float(np.sqrt(np.mean(np.square(frame))))
    zcr = _zero_crossing_rate(frame)
    return energy * zcr
