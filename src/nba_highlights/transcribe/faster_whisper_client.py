from nba_highlights.interfaces import Transcript, TranscriptSegment, Word


class FasterWhisperTranscriber:
    def __init__(self, model_size: str = "small", device: str = "cpu",
                 compute_type: str = "int8") -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._model = None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self._model_size, device=self._device,
                                       compute_type=self._compute_type)
        return self._model

    def transcribe(self, audio_path: str) -> Transcript:
        model = self._load()
        segments, _info = model.transcribe(audio_path, word_timestamps=True)
        out: list[TranscriptSegment] = []
        for seg in segments:
            words = [Word(text=w.word.strip(), start_s=w.start, end_s=w.end)
                     for w in (seg.words or [])]
            out.append(TranscriptSegment(start_s=seg.start, end_s=seg.end,
                                         text=seg.text.strip(), words=words))
        return Transcript(segments=out)
