from nba_highlights.interfaces import ActionResult
from nba_highlights.action.basketball import score_from_probs
from nba_highlights.media import frame_times


class VideoMAEActionRecognizer:
    def __init__(self, model_name: str = "MCG-NJU/videomae-base-finetuned-kinetics",
                 device: str = "cpu", num_frames: int = 16) -> None:
        self._model_name = model_name
        self._device = device
        self._num_frames = num_frames
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
            self._processor = VideoMAEImageProcessor.from_pretrained(self._model_name)
            self._model = VideoMAEForVideoClassification.from_pretrained(
                self._model_name).to(self._device).eval()
            self._torch = torch
        return self._model, self._processor

    def _read_frames(self, video_path: str, start_s: float, end_s: float):
        # Seek to start_s by timestamp and decode only frames inside the clip
        # window, then pick the frame nearest each target sample time. Do NOT
        # match on frame.index: that counts decoded frames from the file start,
        # so a clip at t=1400s would (wrongly, and slowly) decode from 0.
        import av
        import numpy as np
        targets = frame_times(start_s, end_s, self._num_frames)
        container = av.open(video_path)
        stream = container.streams.video[0]
        if stream.time_base:
            container.seek(int(start_s / stream.time_base), stream=stream,
                           backward=True)
        collected: list[tuple[float, "np.ndarray"]] = []
        for frame in container.decode(video=0):
            t = float(frame.time) if frame.time is not None else 0.0
            if t < start_s:
                continue
            if t > end_s:
                break
            collected.append((t, np.asarray(frame.to_ndarray(format="rgb24"))))
        container.close()
        if not collected:
            return []
        # For each evenly-spaced target time, take the nearest decoded frame.
        return [min(collected, key=lambda ct: abs(ct[0] - target))[1]
                for target in targets]

    def score_clip(self, video_path: str, start_s: float,
                   end_s: float) -> ActionResult:
        model, processor = self._load()
        frames = self._read_frames(video_path, start_s, end_s)
        if not frames:
            return ActionResult(score=0.0, label="")
        inputs = processor(frames, return_tensors="pt").to(self._device)
        with self._torch.no_grad():
            logits = model(**inputs).logits[0]
            probs = self._torch.softmax(logits, dim=-1)
        id2label = model.config.id2label
        label_probs = {id2label[i]: float(probs[i]) for i in range(len(probs))}
        return score_from_probs(label_probs)
