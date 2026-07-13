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
        import av
        import numpy as np
        times = frame_times(start_s, end_s, self._num_frames)
        container = av.open(video_path)
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 25)
        frames: list = []
        wanted = [int(t * fps) for t in times]
        for frame in container.decode(video=0):
            if frame.index in wanted:
                frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= self._num_frames:
                break
        container.close()
        while len(frames) < self._num_frames and frames:
            frames.append(frames[-1])
        return [np.asarray(f) for f in frames]

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
