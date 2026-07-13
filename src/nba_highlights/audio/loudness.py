import numpy as np


def rms_envelope(samples: np.ndarray, sr: int, hop_s: float = 0.05,
                 win_s: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    hop = max(1, int(sr * hop_s))
    win = max(1, int(sr * win_s))
    times: list[float] = []
    rms: list[float] = []
    for start in range(0, len(samples), hop):
        frame = samples[start:start + win]
        if len(frame) == 0:
            break
        rms.append(float(np.sqrt(np.mean(np.square(frame)))))
        times.append(start / sr)
    return np.array(times), np.array(rms)


def shot_loudness(times: np.ndarray, rms: np.ndarray,
                  start_s: float, end_s: float) -> tuple[float, float | None]:
    mask = (times >= start_s) & (times < end_s)
    if not mask.any():
        return 0.0, None
    window = rms[mask]
    window_times = times[mask]
    idx = int(np.argmax(window))
    return float(window[idx]), float(window_times[idx])


def load_audio(wav_path: str) -> tuple[np.ndarray, int]:
    import soundfile as sf
    samples, sr = sf.read(wav_path)
    if samples.ndim > 1:  # stereo -> mono
        samples = samples.mean(axis=1)
    return samples.astype(float), int(sr)
