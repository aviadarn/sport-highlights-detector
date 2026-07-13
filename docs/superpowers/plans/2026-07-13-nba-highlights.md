# NBA Highlights Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a YouTube full-game NBA URL into a JSON report of play-level highlight segments with timestamps, a fused audio+action score, and per-signal evidence.

**Architecture:** Linear in-process pipeline: ingest (yt-dlp + ffmpeg) → scene detect (PySceneDetect) → per-shot audio scoring (loudness + prosody + ASR keyword spotting) → two-pass action recognition (video model on audio-gated candidate shots) → weighted fusion → merge adjacent high-scoring shots into play segments → threshold select → Report JSON. Every external dependency (transcriber, action recognizer, downloader, ffmpeg runner, scene detector) sits behind an injectable seam so the whole pipeline runs deterministically in unit tests with stub backends and no model downloads.

**Tech Stack:** Python 3.11+, pydantic v2, typer, numpy. Optional heavy extras (lazy-imported, never needed for tests): yt-dlp + scenedetect (`media`), librosa (`audio`), faster-whisper (`asr`), torch + transformers/VideoMAE (`action`). Test/lint: pytest + ruff.

## Global Constraints

- Python `>=3.11` (developed on 3.13). Copied verbatim from spec §8.
- Package name: `nba_highlights`. Source under `src/nba_highlights/`, layout in spec §8.
- All heavy ML/media libraries are **lazy-imported inside functions**, never at module top level. `pip install -e '.[dev]'` alone must run the entire test suite (stub backends only, no network, no model downloads).
- Every backend has a deterministic **stub** default. Default `action_backend="stub"`, `asr_backend="stub"`.
- Fusion weights: `audio_composite = wL·loudness + wP·prosody + wK·keywords` with `wL+wP+wK=1`; `fused = wA·audio_composite + wV·action` with `wA+wV=1`. Defaults `wL/wP/wK = 0.4/0.3/0.3`, `wA/wV = 0.5/0.5`. All signal values are percentile-normalized to 0–1 per game before fusion.
- Selection: `fused ≥ threshold` (default 0.5). Two-pass action gate: action model runs only on shots with `audio_composite ≥ audio_gate` (default 0.35) unless `full_action=True`.
- Merge: adjacent qualifying shots with inter-shot time gap `≤ max_gap_s` (default 3.0) collapse into one segment; segment score = max of member shot fused scores.
- `slow` pytest marker deselected by default (`addopts = "-m 'not slow'"`); real-model integration tests carry it.
- ruff `line-length = 100`, `target-version = "py311"`.

---

### Task 1: Project scaffold + error types

**Files:**
- Create: `pyproject.toml`
- Create: `src/nba_highlights/__init__.py`
- Create: `src/nba_highlights/errors.py`
- Create: `tests/__init__.py`
- Create: `tests/test_version.py`
- Create: `.gitignore`

**Interfaces:**
- Consumes: nothing.
- Produces: package `nba_highlights` with `__version__: str`; `nba_highlights.errors.IngestError`, `nba_highlights.errors.ConfigError` (both subclass `Exception`).

- [ ] **Step 1: Write `.gitignore`**

```gitignore
__pycache__/
*.pyc
.venv/
.pytest_cache/
.ruff_cache/
*.egg-info/
/work/
/data/work/
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "nba-highlights"
version = "0.1.0"
description = "NBA video highlight detection (audio cues + action recognition) -> JSON report"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.6",
    "typer>=0.12",
    "numpy>=1.26",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.4"]
# Heavy runtime backends — lazy-imported, only needed for real (non-test) runs.
media = ["scenedetect>=0.6.3", "yt-dlp>=2024.4.9"]
audio = ["librosa>=0.10", "soundfile>=0.12"]
asr = ["faster-whisper>=1.0"]
action = ["torch>=2.2", "transformers>=4.40", "av>=12.0"]

[project.scripts]
highlights = "nba_highlights.cli:app"

[tool.hatch.build.targets.wheel]
packages = ["src/nba_highlights"]

[tool.pytest.ini_options]
pythonpath = ["src"]
markers = [
    "slow: requires real models or network (deselected by default)",
]
addopts = "-m 'not slow'"

[tool.ruff]
line-length = 100
target-version = "py311"
```

- [ ] **Step 3: Write `src/nba_highlights/__init__.py`**

```python
__version__ = "0.1.0"
```

- [ ] **Step 4: Write `src/nba_highlights/errors.py`**

```python
class IngestError(Exception):
    """Raised when downloading or extracting media fails."""


class ConfigError(Exception):
    """Raised on invalid configuration (bad weights, unknown backend)."""
```

- [ ] **Step 5: Write `tests/__init__.py` (empty) and `tests/test_version.py`**

```python
import nba_highlights
from nba_highlights.errors import ConfigError, IngestError


def test_version_is_string():
    assert isinstance(nba_highlights.__version__, str)


def test_errors_are_exceptions():
    assert issubclass(IngestError, Exception)
    assert issubclass(ConfigError, Exception)
```

- [ ] **Step 6: Install and run tests**

Run:
```bash
python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest -q
```
Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src tests .gitignore
git commit -m "feat: project scaffold + error types"
```

---

### Task 2: Domain models

**Files:**
- Create: `src/nba_highlights/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: nothing.
- Produces (all pydantic `BaseModel`):
  - `VideoInfo{source:str, title:str="", duration_s:float=0.0}`
  - `Shot{scene_id:int, start_s:float, end_s:float}`
  - `AudioSignals{loudness:float=0.0, prosody:float=0.0, keywords:float=0.0}`
  - `ActionSignals{score:float=0.0, label:str=""}`
  - `ShotScore{scene_id:int, start_s:float, end_s:float, audio:AudioSignals, action:ActionSignals, audio_composite:float=0.0, fused:float=0.0, matched_keywords:list[str]=[], peak_loudness_s:float|None=None, transcript:str=""}`
  - `SignalBreakdown{audio:AudioSignals, action:ActionSignals}`
  - `Evidence{peak_loudness_s:float|None=None, keywords:list[str]=[], transcript:str=""}`
  - `Highlight{id:int, start_s:float, end_s:float, score:float, signals:SignalBreakdown, evidence:Evidence, member_scenes:list[int]}`
  - `Report{video:VideoInfo, config:dict, generated_at:str, highlights:list[Highlight]}`

- [ ] **Step 1: Write the failing test**

```python
from nba_highlights.models import (
    VideoInfo, Shot, AudioSignals, ActionSignals, ShotScore,
    SignalBreakdown, Evidence, Highlight, Report,
)


def test_shot_score_defaults():
    s = ShotScore(scene_id=3, start_s=10.0, end_s=12.0,
                  audio=AudioSignals(loudness=0.9), action=ActionSignals())
    assert s.fused == 0.0
    assert s.matched_keywords == []
    assert s.peak_loudness_s is None


def test_report_round_trip():
    r = Report(
        video=VideoInfo(source="u", title="t", duration_s=100.0),
        config={"threshold": 0.5},
        generated_at="2026-07-13T00:00:00Z",
        highlights=[Highlight(
            id=0, start_s=10.0, end_s=13.0, score=0.8,
            signals=SignalBreakdown(audio=AudioSignals(loudness=0.9),
                                    action=ActionSignals(score=0.7, label="dunking basketball")),
            evidence=Evidence(peak_loudness_s=11.0, keywords=["dunk"], transcript="oh my"),
            member_scenes=[4, 5],
        )],
    )
    dumped = r.model_dump_json()
    back = Report.model_validate_json(dumped)
    assert back.highlights[0].signals.action.label == "dunking basketball"
    assert back.highlights[0].member_scenes == [4, 5]
    assert back.video.duration_s == 100.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nba_highlights.models'`.

- [ ] **Step 3: Write `src/nba_highlights/models.py`**

```python
from pydantic import BaseModel


class VideoInfo(BaseModel):
    source: str
    title: str = ""
    duration_s: float = 0.0


class Shot(BaseModel):
    scene_id: int
    start_s: float
    end_s: float


class AudioSignals(BaseModel):
    loudness: float = 0.0
    prosody: float = 0.0
    keywords: float = 0.0


class ActionSignals(BaseModel):
    score: float = 0.0
    label: str = ""


class ShotScore(BaseModel):
    scene_id: int
    start_s: float
    end_s: float
    audio: AudioSignals
    action: ActionSignals
    audio_composite: float = 0.0
    fused: float = 0.0
    matched_keywords: list[str] = []
    peak_loudness_s: float | None = None
    transcript: str = ""


class SignalBreakdown(BaseModel):
    audio: AudioSignals
    action: ActionSignals


class Evidence(BaseModel):
    peak_loudness_s: float | None = None
    keywords: list[str] = []
    transcript: str = ""


class Highlight(BaseModel):
    id: int
    start_s: float
    end_s: float
    score: float
    signals: SignalBreakdown
    evidence: Evidence
    member_scenes: list[int]


class Report(BaseModel):
    video: VideoInfo
    config: dict
    generated_at: str
    highlights: list[Highlight]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_models.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nba_highlights/models.py tests/test_models.py
git commit -m "feat: domain models for shots, signals, highlights, report"
```

---

### Task 3: Interfaces (Transcript + protocols)

**Files:**
- Create: `src/nba_highlights/interfaces.py`
- Test: `tests/test_interfaces.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Word{text:str, start_s:float, end_s:float}`
  - `TranscriptSegment{start_s:float, end_s:float, text:str, words:list[Word]}`
  - `Transcript{segments:list[TranscriptSegment]}` with `.duration_s -> float`, `.words_in(start_s,end_s) -> list[Word]` (word included if its midpoint falls in `[start_s, end_s)`), `.text_in(start_s,end_s) -> str`
  - `ActionResult{score:float, label:str, top_k:list[tuple[str,float]]=[]}`
  - `Transcriber` protocol: `transcribe(audio_path:str) -> Transcript`
  - `ActionRecognizer` protocol: `score_clip(video_path:str, start_s:float, end_s:float) -> ActionResult`

- [ ] **Step 1: Write the failing test**

```python
from nba_highlights.interfaces import Word, TranscriptSegment, Transcript, ActionResult


def _transcript():
    return Transcript(segments=[TranscriptSegment(
        start_s=0.0, end_s=4.0, text="what a dunk oh my",
        words=[Word(text="what", start_s=0.0, end_s=0.5),
               Word(text="a", start_s=0.5, end_s=0.8),
               Word(text="dunk", start_s=1.0, end_s=1.4),
               Word(text="oh", start_s=3.0, end_s=3.2),
               Word(text="my", start_s=3.2, end_s=3.5)])])


def test_words_in_uses_midpoint_window():
    t = _transcript()
    picked = [w.text for w in t.words_in(0.9, 2.0)]
    assert picked == ["dunk"]


def test_text_in_joins_words():
    t = _transcript()
    assert t.text_in(2.5, 4.0) == "oh my"


def test_duration_is_last_segment_end():
    assert _transcript().duration_s == 4.0
    assert Transcript(segments=[]).duration_s == 0.0


def test_action_result_defaults():
    r = ActionResult(score=0.5, label="dunking basketball")
    assert r.top_k == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_interfaces.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/nba_highlights/interfaces.py`**

```python
from typing import Protocol
from pydantic import BaseModel


class Word(BaseModel):
    text: str
    start_s: float
    end_s: float


class TranscriptSegment(BaseModel):
    start_s: float
    end_s: float
    text: str
    words: list[Word]


class Transcript(BaseModel):
    segments: list[TranscriptSegment]

    @property
    def duration_s(self) -> float:
        return self.segments[-1].end_s if self.segments else 0.0

    def words_in(self, start_s: float, end_s: float) -> list[Word]:
        out: list[Word] = []
        for seg in self.segments:
            for w in seg.words:
                mid = (w.start_s + w.end_s) / 2.0
                if start_s <= mid < end_s:
                    out.append(w)
        return out

    def text_in(self, start_s: float, end_s: float) -> str:
        return " ".join(w.text for w in self.words_in(start_s, end_s)).strip()


class ActionResult(BaseModel):
    score: float
    label: str
    top_k: list[tuple[str, float]] = []


class Transcriber(Protocol):
    def transcribe(self, audio_path: str) -> Transcript: ...


class ActionRecognizer(Protocol):
    def score_clip(self, video_path: str, start_s: float, end_s: float) -> ActionResult: ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_interfaces.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nba_highlights/interfaces.py tests/test_interfaces.py
git commit -m "feat: Transcript model + Transcriber/ActionRecognizer protocols"
```

---

### Task 4: Settings / config

**Files:**
- Create: `src/nba_highlights/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: `nba_highlights.errors.ConfigError`.
- Produces:
  - `Weights{wL:float=0.4, wP:float=0.3, wK:float=0.3, wA:float=0.5, wV:float=0.5}`
  - `DEFAULT_HYPE_LEXICON: list[str]`
  - `ACTION_BACKENDS: set[str] = {"stub","videomae"}`, `ASR_BACKENDS: set[str] = {"stub","faster-whisper"}`
  - `Settings{weights:Weights, threshold:float=0.5, audio_gate:float=0.35, max_gap_s:float=3.0, action_backend:str="stub", asr_backend:str="stub", asr_model:str="small", full_action:bool=False, hype_lexicon:list[str]}` with classmethod `from_env() -> Settings` and method `check() -> None` (raises `ConfigError` on unknown backend or on weight groups not summing to 1.0 within 1e-6). NOTE: named `check`, not `validate`, to avoid shadowing pydantic v2's deprecated `BaseModel.validate` classmethod.

- [ ] **Step 1: Write the failing test**

```python
import pytest
from nba_highlights.config import Settings, Weights
from nba_highlights.errors import ConfigError


def test_defaults():
    s = Settings()
    assert s.action_backend == "stub"
    assert s.asr_backend == "stub"
    assert s.threshold == 0.5
    assert s.audio_gate == 0.35
    assert s.max_gap_s == 3.0
    assert "dunk" in s.hype_lexicon


def test_validate_rejects_unknown_backend():
    with pytest.raises(ConfigError):
        Settings(action_backend="nope").check()
    with pytest.raises(ConfigError):
        Settings(asr_backend="nope").check()


def test_validate_rejects_bad_weight_sum():
    with pytest.raises(ConfigError):
        Settings(weights=Weights(wL=0.5, wP=0.5, wK=0.5)).check()


def test_from_env_overrides(monkeypatch):
    monkeypatch.setenv("ACTION_BACKEND", "videomae")
    monkeypatch.setenv("ASR_BACKEND", "faster-whisper")
    monkeypatch.setenv("THRESHOLD", "0.7")
    s = Settings.from_env()
    assert s.action_backend == "videomae"
    assert s.asr_backend == "faster-whisper"
    assert s.threshold == 0.7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/nba_highlights/config.py`**

```python
import os
from pydantic import BaseModel
from nba_highlights.errors import ConfigError

DEFAULT_HYPE_LEXICON = [
    "dunk", "dunks", "slam", "three", "threes", "and one", "and-one",
    "buzzer", "oh my", "are you kidding", "posterized", "block", "blocked",
    "steal", "alley oop", "alley-oop", "what a", "unbelievable", "clutch",
    "dagger", "and the foul", "from downtown", "count it",
]

ACTION_BACKENDS = {"stub", "videomae"}
ASR_BACKENDS = {"stub", "faster-whisper"}


class Weights(BaseModel):
    wL: float = 0.4
    wP: float = 0.3
    wK: float = 0.3
    wA: float = 0.5
    wV: float = 0.5


class Settings(BaseModel):
    weights: Weights = Weights()
    threshold: float = 0.5
    audio_gate: float = 0.35
    max_gap_s: float = 3.0
    action_backend: str = "stub"
    asr_backend: str = "stub"
    asr_model: str = "small"
    full_action: bool = False
    hype_lexicon: list[str] = DEFAULT_HYPE_LEXICON

    def check(self) -> None:
        if self.action_backend not in ACTION_BACKENDS:
            raise ConfigError(f"unknown action_backend: {self.action_backend!r}")
        if self.asr_backend not in ASR_BACKENDS:
            raise ConfigError(f"unknown asr_backend: {self.asr_backend!r}")
        w = self.weights
        if abs((w.wL + w.wP + w.wK) - 1.0) > 1e-6:
            raise ConfigError("audio weights wL+wP+wK must sum to 1.0")
        if abs((w.wA + w.wV) - 1.0) > 1e-6:
            raise ConfigError("fusion weights wA+wV must sum to 1.0")

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls()
        s.action_backend = os.environ.get("ACTION_BACKEND", s.action_backend)
        s.asr_backend = os.environ.get("ASR_BACKEND", s.asr_backend)
        s.asr_model = os.environ.get("ASR_MODEL", s.asr_model)
        if "THRESHOLD" in os.environ:
            s.threshold = float(os.environ["THRESHOLD"])
        if "AUDIO_GATE" in os.environ:
            s.audio_gate = float(os.environ["AUDIO_GATE"])
        if "MAX_GAP_S" in os.environ:
            s.max_gap_s = float(os.environ["MAX_GAP_S"])
        s.check()
        return s
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nba_highlights/config.py tests/test_config.py
git commit -m "feat: Settings with weights, gates, backend validation, env override"
```

---

### Task 5: Media ingest + frame sampling

**Files:**
- Create: `src/nba_highlights/media.py`
- Test: `tests/test_media.py`

**Interfaces:**
- Consumes: `nba_highlights.errors.IngestError`.
- Produces:
  - `classify_source(locator:str) -> str` returning `"youtube"|"hls"|"file"`
  - `ingest(locator:str, workdir:str, downloader=_default_downloader, runner=subprocess.run) -> tuple[str,str]` → `(video_path, audio_path)`; extracts 16 kHz mono WAV `audio.wav`; wraps failures in `IngestError`
  - `probe_duration(video_path:str, runner=subprocess.run) -> float`
  - `frame_times(start_s:float, end_s:float, n:int) -> list[float]` (pure; `n` evenly spaced fractions `(j+1)/(n+1)` across the span; `n>=1`)
  - `extract_frame(video_path:str, t_s:float, out_path:str, runner=subprocess.run) -> str`

- [ ] **Step 1: Write the failing test**

```python
import subprocess
import pytest
from nba_highlights.media import (
    classify_source, ingest, probe_duration, frame_times, extract_frame,
)
from nba_highlights.errors import IngestError


def test_classify_source():
    assert classify_source("https://www.youtube.com/watch?v=x") == "youtube"
    assert classify_source("https://youtu.be/x") == "youtube"
    assert classify_source("https://cdn.example.com/stream.m3u8") == "hls"
    assert classify_source("/tmp/game.mp4") == "file"


def test_frame_times_evenly_spaced():
    assert frame_times(0.0, 10.0, 1) == [5.0]
    assert frame_times(0.0, 12.0, 3) == [3.0, 6.0, 9.0]
    assert frame_times(4.0, 4.0, 2) == [4.0, 4.0]  # zero-span is safe


def test_ingest_file_runs_ffmpeg_for_audio(tmp_path):
    calls = []

    def fake_runner(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    video, audio = ingest("/videos/game.mp4", str(tmp_path), runner=fake_runner)
    assert video == "/videos/game.mp4"
    assert audio.endswith("audio.wav")
    assert calls[0][0] == "ffmpeg"
    assert "16000" in calls[0]


def test_ingest_download_failure_raises_ingest_error(tmp_path):
    def boom(url, out_template):
        raise RuntimeError("network down")

    with pytest.raises(IngestError):
        ingest("https://youtu.be/x", str(tmp_path), downloader=boom)


def test_extract_frame_invokes_ffmpeg(tmp_path):
    calls = []
    out = str(tmp_path / "f.jpg")
    extract_frame("/v.mp4", 12.5, out,
                  runner=lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0))
    assert calls[0][0] == "ffmpeg" and out in calls[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_media.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/nba_highlights/media.py`**

```python
import os
import subprocess
from urllib.parse import urlparse
from nba_highlights.errors import IngestError


def classify_source(locator: str) -> str:
    host = (urlparse(locator).netloc or "").lower()
    if "youtube.com" in host or "youtu.be" in host:
        return "youtube"
    if locator.lower().endswith(".m3u8"):
        return "hls"
    return "file"


def _default_downloader(url: str, out_template: str) -> str:
    from yt_dlp import YoutubeDL
    opts = {"format": "mp4/best", "outtmpl": out_template, "quiet": True,
            "noplaylist": True}
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)


def ingest(locator: str, workdir: str, downloader=_default_downloader,
           runner=subprocess.run) -> tuple[str, str]:
    os.makedirs(workdir, exist_ok=True)
    kind = classify_source(locator)
    try:
        if kind == "file":
            video_path = locator
        else:
            video_path = downloader(locator, os.path.join(workdir, "video.%(ext)s"))
        audio_path = os.path.join(workdir, "audio.wav")
        runner(["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1",
                "-ar", "16000", audio_path], check=True)
    except IngestError:
        raise
    except Exception as e:  # download/ffmpeg failure -> uniform error
        raise IngestError(f"ingest failed for {locator!r}: {e}") from e
    return video_path, audio_path


def probe_duration(video_path: str, runner=subprocess.run) -> float:
    try:
        result = runner(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def frame_times(start_s: float, end_s: float, n: int) -> list[float]:
    n = max(1, n)
    span = max(0.0, end_s - start_s)
    return [start_s + (j + 1) / (n + 1) * span for j in range(n)]


def extract_frame(video_path: str, t_s: float, out_path: str,
                  runner=subprocess.run) -> str:
    runner(["ffmpeg", "-y", "-ss", str(t_s), "-i", video_path,
            "-frames:v", "1", "-q:v", "2", out_path], check=True)
    return out_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_media.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nba_highlights/media.py tests/test_media.py
git commit -m "feat: media ingest (yt-dlp+ffmpeg), duration probe, frame sampling"
```

---

### Task 6: Scene detection

**Files:**
- Create: `src/nba_highlights/scenes.py`
- Test: `tests/test_scenes.py`

**Interfaces:**
- Consumes: `nba_highlights.models.Shot`.
- Produces: `detect_scenes(video_path:str, duration_s:float, detector_fn=_default_detector) -> list[Shot]`. `detector_fn(video_path) -> list[tuple[float,float]]`. Empty detector output → single shot `(0.0, duration_s)`. Shots get sequential `scene_id` from 0.

- [ ] **Step 1: Write the failing test**

```python
from nba_highlights.scenes import detect_scenes
from nba_highlights.models import Shot


def test_detect_scenes_maps_boundaries_to_shots():
    shots = detect_scenes("v.mp4", 30.0,
                          detector_fn=lambda p: [(0.0, 5.0), (5.0, 12.0)])
    assert shots == [Shot(scene_id=0, start_s=0.0, end_s=5.0),
                     Shot(scene_id=1, start_s=5.0, end_s=12.0)]


def test_detect_scenes_empty_is_single_shot_over_duration():
    shots = detect_scenes("v.mp4", 42.0, detector_fn=lambda p: [])
    assert shots == [Shot(scene_id=0, start_s=0.0, end_s=42.0)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_scenes.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/nba_highlights/scenes.py`**

```python
from nba_highlights.models import Shot


def _default_detector(video_path: str) -> list[tuple[float, float]]:
    from scenedetect import detect, AdaptiveDetector
    scene_list = detect(video_path, AdaptiveDetector())
    return [(s.get_seconds(), e.get_seconds()) for s, e in scene_list]


def detect_scenes(video_path: str, duration_s: float,
                  detector_fn=_default_detector) -> list[Shot]:
    boundaries = detector_fn(video_path)
    if not boundaries:
        boundaries = [(0.0, duration_s)]
    return [Shot(scene_id=i, start_s=s, end_s=e)
            for i, (s, e) in enumerate(boundaries)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_scenes.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nba_highlights/scenes.py tests/test_scenes.py
git commit -m "feat: PySceneDetect scene detection -> Shot list"
```

---

### Task 7: Transcriber backends (stub + faster-whisper)

**Files:**
- Create: `src/nba_highlights/transcribe/__init__.py`
- Create: `src/nba_highlights/transcribe/stub.py`
- Create: `src/nba_highlights/transcribe/faster_whisper_client.py`
- Test: `tests/test_transcribe.py`

**Interfaces:**
- Consumes: `nba_highlights.interfaces.{Transcript,TranscriptSegment,Word}`.
- Produces:
  - `StubTranscriber(transcript: Transcript | None = None)` — `.transcribe(audio_path)` returns the injected transcript, or an empty `Transcript(segments=[])` if none. Satisfies the `Transcriber` protocol.
  - `FasterWhisperTranscriber(model_size:str="small", device:str="cpu", compute_type:str="int8")` — `.transcribe(audio_path)` lazy-loads faster-whisper and maps its output to `Transcript`.

- [ ] **Step 1: Write the failing test**

```python
from nba_highlights.transcribe.stub import StubTranscriber
from nba_highlights.interfaces import Transcript, TranscriptSegment, Word


def test_stub_returns_empty_by_default():
    assert StubTranscriber().transcribe("a.wav").segments == []


def test_stub_returns_injected_transcript():
    t = Transcript(segments=[TranscriptSegment(
        start_s=0.0, end_s=1.0, text="dunk",
        words=[Word(text="dunk", start_s=0.0, end_s=0.5)])])
    assert StubTranscriber(t).transcribe("a.wav").text_in(0.0, 1.0) == "dunk"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_transcribe.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the three files**

`src/nba_highlights/transcribe/__init__.py`:
```python
```

`src/nba_highlights/transcribe/stub.py`:
```python
from nba_highlights.interfaces import Transcript


class StubTranscriber:
    def __init__(self, transcript: Transcript | None = None) -> None:
        self._transcript = transcript or Transcript(segments=[])

    def transcribe(self, audio_path: str) -> Transcript:
        return self._transcript
```

`src/nba_highlights/transcribe/faster_whisper_client.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_transcribe.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nba_highlights/transcribe tests/test_transcribe.py
git commit -m "feat: stub + faster-whisper transcriber backends"
```

---

### Task 8: Audio — loudness scoring

**Files:**
- Create: `src/nba_highlights/audio/__init__.py`
- Create: `src/nba_highlights/audio/loudness.py`
- Test: `tests/test_loudness.py`

**Interfaces:**
- Consumes: nothing (operates on numpy arrays).
- Produces:
  - `rms_envelope(samples:np.ndarray, sr:int, hop_s:float=0.05, win_s:float=0.05) -> tuple[np.ndarray, np.ndarray]` → `(times, rms)` arrays, one entry per hop.
  - `shot_loudness(times:np.ndarray, rms:np.ndarray, start_s:float, end_s:float) -> tuple[float, float | None]` → `(peak_rms_in_shot, time_of_peak)`; returns `(0.0, None)` if no envelope samples fall in the shot.
  - `load_audio(wav_path:str) -> tuple[np.ndarray, int]` (lazy-imports soundfile; returns mono samples + sample rate).

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from nba_highlights.audio.loudness import rms_envelope, shot_loudness


def test_rms_envelope_shapes_align():
    sr = 1000
    samples = np.ones(sr * 2, dtype=float)  # 2 seconds of constant signal
    times, rms = rms_envelope(samples, sr, hop_s=0.1, win_s=0.1)
    assert len(times) == len(rms)
    assert np.allclose(rms, 1.0, atol=1e-6)


def test_shot_loudness_finds_peak_in_window():
    sr = 1000
    samples = np.zeros(sr * 3, dtype=float)
    samples[sr * 1 : sr * 1 + 100] = 5.0  # loud burst at ~1.0s
    times, rms = rms_envelope(samples, sr, hop_s=0.05, win_s=0.05)
    peak, t_peak = shot_loudness(times, rms, 0.5, 1.5)
    assert peak > 0.0
    assert 0.9 <= t_peak <= 1.2


def test_shot_loudness_empty_window():
    times = np.array([0.0, 0.1, 0.2])
    rms = np.array([1.0, 1.0, 1.0])
    assert shot_loudness(times, rms, 5.0, 6.0) == (0.0, None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_loudness.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the files**

`src/nba_highlights/audio/__init__.py`:
```python
```

`src/nba_highlights/audio/loudness.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_loudness.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nba_highlights/audio/__init__.py src/nba_highlights/audio/loudness.py tests/test_loudness.py
git commit -m "feat: audio loudness scoring (RMS envelope + per-shot peak)"
```

---

### Task 9: Audio — prosody (commentator excitement)

**Files:**
- Create: `src/nba_highlights/audio/prosody.py`
- Test: `tests/test_prosody.py`

**Interfaces:**
- Consumes: nothing (numpy arrays).
- Produces:
  - `shot_prosody(samples:np.ndarray, sr:int, start_s:float, end_s:float) -> float` — raw excitement proxy for a shot: mean short-time energy times spectral-centroid rise. Deterministic, no model. Returns `0.0` for an empty/zero window. Implementation uses a pure-numpy energy × zero-crossing-rate product so it needs no heavy deps and is unit-testable; louder + higher-frequency (excited speech) windows score higher.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from nba_highlights.audio.prosody import shot_prosody


def test_prosody_zero_for_silence():
    sr = 1000
    samples = np.zeros(sr * 2, dtype=float)
    assert shot_prosody(samples, sr, 0.0, 2.0) == 0.0


def test_prosody_higher_for_loud_high_freq():
    sr = 2000
    t = np.arange(sr * 2) / sr
    calm = 0.1 * np.sin(2 * np.pi * 100 * t)      # quiet, low freq
    excited = 1.0 * np.sin(2 * np.pi * 500 * t)   # loud, high freq
    calm_score = shot_prosody(calm, sr, 0.0, 2.0)
    excited_score = shot_prosody(excited, sr, 0.0, 2.0)
    assert excited_score > calm_score


def test_prosody_empty_window_is_zero():
    sr = 1000
    samples = np.ones(sr, dtype=float)
    assert shot_prosody(samples, sr, 5.0, 6.0) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_prosody.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/nba_highlights/audio/prosody.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_prosody.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nba_highlights/audio/prosody.py tests/test_prosody.py
git commit -m "feat: audio prosody scoring (energy x zero-crossing excitement proxy)"
```

---

### Task 10: Audio — keyword spotting

**Files:**
- Create: `src/nba_highlights/audio/keywords.py`
- Test: `tests/test_keywords.py`

**Interfaces:**
- Consumes: `nba_highlights.interfaces.Transcript`.
- Produces:
  - `shot_keywords(transcript:Transcript, start_s:float, end_s:float, lexicon:list[str]) -> tuple[float, list[str]]` → `(raw_score, matched_phrases)`. Match is case-insensitive substring of the shot's joined transcript text (`transcript.text_in`). Multi-word phrases ("and one") match across word boundaries. `raw_score = number of distinct matched phrases`. Returns `(0.0, [])` when no words fall in the shot.

- [ ] **Step 1: Write the failing test**

```python
from nba_highlights.audio.keywords import shot_keywords
from nba_highlights.interfaces import Transcript, TranscriptSegment, Word

LEX = ["dunk", "and one", "three"]


def _t(words):
    return Transcript(segments=[TranscriptSegment(
        start_s=words[0][1], end_s=words[-1][2], text=" ".join(w[0] for w in words),
        words=[Word(text=w[0], start_s=w[1], end_s=w[2]) for w in words])])


def test_matches_single_and_multiword():
    t = _t([("what", 0.0, 0.3), ("a", 0.3, 0.5), ("dunk", 0.6, 1.0),
            ("and", 1.1, 1.3), ("one", 1.3, 1.6)])
    score, matched = shot_keywords(t, 0.0, 2.0, LEX)
    assert set(matched) == {"dunk", "and one"}
    assert score == 2.0


def test_no_words_in_window():
    t = _t([("dunk", 0.0, 0.5)])
    assert shot_keywords(t, 10.0, 11.0, LEX) == (0.0, [])


def test_case_insensitive():
    t = _t([("DUNK", 0.0, 0.5)])
    score, matched = shot_keywords(t, 0.0, 1.0, LEX)
    assert matched == ["dunk"] and score == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_keywords.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/nba_highlights/audio/keywords.py`**

```python
from nba_highlights.interfaces import Transcript


def shot_keywords(transcript: Transcript, start_s: float, end_s: float,
                  lexicon: list[str]) -> tuple[float, list[str]]:
    text = transcript.text_in(start_s, end_s).lower()
    if not text:
        return 0.0, []
    matched = [phrase for phrase in lexicon if phrase.lower() in text]
    return float(len(matched)), matched
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_keywords.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nba_highlights/audio/keywords.py tests/test_keywords.py
git commit -m "feat: audio keyword spotting against hype lexicon"
```

---

### Task 11: Action recognizer backends (stub + VideoMAE)

**Files:**
- Create: `src/nba_highlights/action/__init__.py`
- Create: `src/nba_highlights/action/basketball.py`
- Create: `src/nba_highlights/action/stub.py`
- Create: `src/nba_highlights/action/videomae_client.py`
- Test: `tests/test_action.py`

**Interfaces:**
- Consumes: `nba_highlights.interfaces.ActionResult`, `nba_highlights.media.frame_times`.
- Produces:
  - `BASKETBALL_CLASSES: set[str]` — Kinetics-400 labels considered basketball action.
  - `score_from_probs(label_probs:dict[str,float]) -> ActionResult` — `score` = max prob over `BASKETBALL_CLASSES` (0.0 if none present); `label` = the basketball class with that max prob (or `""`); `top_k` = up to 3 highest `(label, prob)` overall sorted desc.
  - `StubActionRecognizer(scores:dict[int,float] | None = None, default:float = 0.0)` — `.score_clip(video, start_s, end_s)` returns `ActionResult(score=scores.get(round(start_s), default), label="stub-basketball" if score>0 else "")`. Deterministic; keyed by `round(start_s)` so tests can target specific shots.
  - `VideoMAEActionRecognizer(model_name:str="MCG-NJU/videomae-base-finetuned-kinetics", device:str="cpu", num_frames:int=16)` — lazy-loads transformers VideoMAE, samples `num_frames` frames across the clip with `frame_times`, runs the model, returns `score_from_probs(...)`.

- [ ] **Step 1: Write the failing test**

```python
from nba_highlights.action.basketball import BASKETBALL_CLASSES, score_from_probs
from nba_highlights.action.stub import StubActionRecognizer


def test_score_from_probs_picks_basketball_class():
    probs = {"dunking basketball": 0.7, "shooting basketball": 0.2, "typing": 0.1}
    r = score_from_probs(probs)
    assert r.score == 0.7
    assert r.label == "dunking basketball"
    assert r.top_k[0] == ("dunking basketball", 0.7)


def test_score_from_probs_zero_when_no_basketball():
    probs = {"typing": 0.9, "reading": 0.1}
    r = score_from_probs(probs)
    assert r.score == 0.0
    assert r.label == ""


def test_basketball_classes_nonempty():
    assert "dunking basketball" in BASKETBALL_CLASSES


def test_stub_action_keyed_by_start_second():
    rec = StubActionRecognizer(scores={12: 0.8}, default=0.1)
    hot = rec.score_clip("v.mp4", 12.2, 14.0)
    cold = rec.score_clip("v.mp4", 30.0, 31.0)
    assert hot.score == 0.8 and hot.label == "stub-basketball"
    assert cold.score == 0.1 and cold.label == "stub-basketball"
    assert rec.score_clip("v.mp4", 40.0, 41.0, ).label == "stub-basketball"


def test_stub_action_zero_label_empty():
    rec = StubActionRecognizer(default=0.0)
    assert rec.score_clip("v.mp4", 1.0, 2.0).label == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_action.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the files**

`src/nba_highlights/action/__init__.py`:
```python
```

`src/nba_highlights/action/basketball.py`:
```python
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
```

`src/nba_highlights/action/stub.py`:
```python
from nba_highlights.interfaces import ActionResult


class StubActionRecognizer:
    def __init__(self, scores: dict[int, float] | None = None,
                 default: float = 0.0) -> None:
        self._scores = scores or {}
        self._default = default

    def score_clip(self, video_path: str, start_s: float,
                   end_s: float) -> ActionResult:
        score = self._scores.get(round(start_s), self._default)
        return ActionResult(score=float(score),
                            label="stub-basketball" if score > 0 else "")
```

`src/nba_highlights/action/videomae_client.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_action.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nba_highlights/action tests/test_action.py
git commit -m "feat: stub + VideoMAE action recognizers, basketball class scoring"
```

---

### Task 12: Fusion + normalization

**Files:**
- Create: `src/nba_highlights/fuse.py`
- Test: `tests/test_fuse.py`

**Interfaces:**
- Consumes: `nba_highlights.config.Weights`.
- Produces:
  - `percentile_normalize(values:list[float]) -> list[float]` — rank each value to `[0,1]` by its fractional rank (`rank / (n-1)`). All-equal input → all `0.0`. Single value → `[0.0]`. Empty → `[]`.
  - `audio_composite(loudness_n:float, prosody_n:float, keywords_n:float, w:Weights) -> float` = `w.wL*loudness_n + w.wP*prosody_n + w.wK*keywords_n`.
  - `fuse(audio_comp:float, action:float, w:Weights) -> float` = `w.wA*audio_comp + w.wV*action`.

- [ ] **Step 1: Write the failing test**

```python
from nba_highlights.fuse import percentile_normalize, audio_composite, fuse
from nba_highlights.config import Weights


def test_percentile_normalize_ranks():
    assert percentile_normalize([10.0, 20.0, 30.0]) == [0.0, 0.5, 1.0]


def test_percentile_normalize_all_equal():
    assert percentile_normalize([5.0, 5.0, 5.0]) == [0.0, 0.0, 0.0]


def test_percentile_normalize_edge_cases():
    assert percentile_normalize([]) == []
    assert percentile_normalize([7.0]) == [0.0]


def test_audio_composite_weighting():
    w = Weights(wL=0.4, wP=0.3, wK=0.3)
    assert abs(audio_composite(1.0, 0.0, 0.0, w) - 0.4) < 1e-9
    assert abs(audio_composite(1.0, 1.0, 1.0, w) - 1.0) < 1e-9


def test_fuse_weighting():
    w = Weights(wA=0.5, wV=0.5)
    assert abs(fuse(0.8, 0.4, w) - 0.6) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_fuse.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/nba_highlights/fuse.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_fuse.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nba_highlights/fuse.py tests/test_fuse.py
git commit -m "feat: percentile normalization + audio/action fusion"
```

---

### Task 13: Merge adjacent shots into segments

**Files:**
- Create: `src/nba_highlights/merge.py`
- Test: `tests/test_merge.py`

**Interfaces:**
- Consumes: `nba_highlights.models.{ShotScore, Highlight, SignalBreakdown, AudioSignals, ActionSignals, Evidence}`.
- Produces: `merge_shots(shot_scores:list[ShotScore], threshold:float, max_gap_s:float) -> list[Highlight]`.
  - Consider only shots with `fused >= threshold`, in start-time order.
  - Start a new group; extend it with the next qualifying shot when `next.start_s - current_group_last.end_s <= max_gap_s`, else close the group and start a new one.
  - Per group → `Highlight`: `start_s`=min member start, `end_s`=max member end, `score`=max member `fused`, `member_scenes`=member `scene_id`s in order. `signals` from the member with the max `fused` (its `audio` + `action`). `evidence`: `peak_loudness_s` from that same peak member, `keywords`=ordered union of members' `matched_keywords` (dedup, preserve first-seen), `transcript`=concatenation of members' non-empty `transcript` joined by `" "`.
  - `id` assigned sequentially from 0 in output order.

- [ ] **Step 1: Write the failing test**

```python
from nba_highlights.merge import merge_shots
from nba_highlights.models import ShotScore, AudioSignals, ActionSignals


def _ss(scene_id, start, end, fused, kw=None, loud=0.0, peak=None, txt=""):
    return ShotScore(
        scene_id=scene_id, start_s=start, end_s=end,
        audio=AudioSignals(loudness=loud), action=ActionSignals(score=fused, label="x"),
        audio_composite=fused, fused=fused, matched_keywords=kw or [],
        peak_loudness_s=peak, transcript=txt)


def test_merges_adjacent_within_gap():
    shots = [_ss(0, 0.0, 2.0, 0.8, kw=["dunk"], peak=1.0, txt="what a dunk"),
             _ss(1, 2.5, 4.0, 0.9, kw=["and one"], peak=3.0, txt="and one"),
             _ss(2, 30.0, 32.0, 0.7, kw=["three"], peak=31.0, txt="from three")]
    out = merge_shots(shots, threshold=0.5, max_gap_s=3.0)
    assert len(out) == 2
    assert out[0].member_scenes == [0, 1]
    assert out[0].start_s == 0.0 and out[0].end_s == 4.0
    assert out[0].score == 0.9              # max fused of the group
    assert out[0].evidence.keywords == ["dunk", "and one"]
    assert out[0].evidence.peak_loudness_s == 3.0   # from the peak member (scene 1)
    assert out[0].id == 0 and out[1].id == 1


def test_below_threshold_shots_excluded():
    shots = [_ss(0, 0.0, 2.0, 0.4), _ss(1, 2.5, 4.0, 0.9)]
    out = merge_shots(shots, threshold=0.5, max_gap_s=3.0)
    assert len(out) == 1 and out[0].member_scenes == [1]


def test_gap_too_large_splits():
    shots = [_ss(0, 0.0, 2.0, 0.8), _ss(1, 10.0, 12.0, 0.8)]
    out = merge_shots(shots, threshold=0.5, max_gap_s=3.0)
    assert len(out) == 2


def test_empty_input():
    assert merge_shots([], threshold=0.5, max_gap_s=3.0) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_merge.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/nba_highlights/merge.py`**

```python
from nba_highlights.models import (
    ShotScore, Highlight, SignalBreakdown, Evidence,
)


def _build_highlight(hid: int, group: list[ShotScore]) -> Highlight:
    peak = max(group, key=lambda s: s.fused)
    keywords: list[str] = []
    for s in group:
        for k in s.matched_keywords:
            if k not in keywords:
                keywords.append(k)
    transcript = " ".join(s.transcript for s in group if s.transcript).strip()
    return Highlight(
        id=hid,
        start_s=min(s.start_s for s in group),
        end_s=max(s.end_s for s in group),
        score=peak.fused,
        signals=SignalBreakdown(audio=peak.audio, action=peak.action),
        evidence=Evidence(peak_loudness_s=peak.peak_loudness_s,
                          keywords=keywords, transcript=transcript),
        member_scenes=[s.scene_id for s in group],
    )


def merge_shots(shot_scores: list[ShotScore], threshold: float,
                max_gap_s: float) -> list[Highlight]:
    qualifying = sorted((s for s in shot_scores if s.fused >= threshold),
                        key=lambda s: s.start_s)
    highlights: list[Highlight] = []
    group: list[ShotScore] = []
    for s in qualifying:
        if group and (s.start_s - group[-1].end_s) <= max_gap_s:
            group.append(s)
        else:
            if group:
                highlights.append(_build_highlight(len(highlights), group))
            group = [s]
    if group:
        highlights.append(_build_highlight(len(highlights), group))
    return highlights
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_merge.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nba_highlights/merge.py tests/test_merge.py
git commit -m "feat: merge adjacent qualifying shots into play-level highlights"
```

---

### Task 14: Select (final threshold + ordering)

**Files:**
- Create: `src/nba_highlights/select.py`
- Test: `tests/test_select.py`

**Interfaces:**
- Consumes: `nba_highlights.models.Highlight`.
- Produces: `select_highlights(highlights:list[Highlight], threshold:float) -> list[Highlight]` — keep highlights with `score >= threshold`, sort by `start_s`, reassign `id` sequentially from 0.

- [ ] **Step 1: Write the failing test**

```python
from nba_highlights.select import select_highlights
from nba_highlights.models import Highlight, SignalBreakdown, AudioSignals, ActionSignals, Evidence


def _hl(hid, start, score):
    return Highlight(id=hid, start_s=start, end_s=start + 1, score=score,
                     signals=SignalBreakdown(audio=AudioSignals(), action=ActionSignals()),
                     evidence=Evidence(), member_scenes=[hid])


def test_filters_and_reorders_and_reids():
    hls = [_hl(0, 30.0, 0.9), _hl(1, 5.0, 0.4), _hl(2, 10.0, 0.6)]
    out = select_highlights(hls, threshold=0.5)
    assert [h.start_s for h in out] == [10.0, 30.0]
    assert [h.id for h in out] == [0, 1]


def test_empty():
    assert select_highlights([], threshold=0.5) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_select.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/nba_highlights/select.py`**

```python
from nba_highlights.models import Highlight


def select_highlights(highlights: list[Highlight], threshold: float) -> list[Highlight]:
    kept = sorted((h for h in highlights if h.score >= threshold),
                  key=lambda h: h.start_s)
    return [h.model_copy(update={"id": i}) for i, h in enumerate(kept)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_select.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nba_highlights/select.py tests/test_select.py
git commit -m "feat: final highlight selection (threshold + order + reindex)"
```

---

### Task 15: Pipeline orchestration (two-pass)

**Files:**
- Create: `src/nba_highlights/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `Settings`, `Transcriber`, `ActionRecognizer`, and all of `media`, `scenes`, `audio.*`, `fuse`, `merge`, `select`, `models`.
- Produces:
```python
def run_pipeline(
    source: str,
    settings: Settings,
    transcriber: Transcriber,
    action: ActionRecognizer,
    *,
    workdir: str,
    generated_at: str,
    title: str = "",
    downloader=media._default_downloader,
    runner=subprocess.run,
    detector_fn=scenes._default_detector,
    audio_loader=loudness.load_audio,
) -> Report
```
  Steps in order: `settings.check()`; `ingest` → `(video, audio)`; `probe_duration`; `detect_scenes`; `transcriber.transcribe(audio)`; `audio_loader(audio)` → `(samples, sr)`; `rms_envelope`; per shot compute raw loudness/prosody/keywords; `percentile_normalize` each of the three raw arrays; compute `audio_composite` per shot; **two-pass gate** — for shots with `audio_composite >= settings.audio_gate` (or all shots if `settings.full_action`) call `action.score_clip`, else action score `0.0`/label `""`; `fuse`; build `ShotScore` list; `merge_shots`; `select_highlights`; assemble `Report` with `VideoInfo`, a `config` dict (`weights`, `threshold`, `audio_gate`, `max_gap_s`, `action_backend`, `asr_backend`, `asr_model`, `full_action`), `generated_at`, `highlights`.
  - Per-shot scoring is wrapped in try/except: a shot that raises during action or audio scoring is logged via `logging.getLogger(__name__).warning(...)` and scored 0, never aborting the run.

- [ ] **Step 1: Write the failing test**

```python
import subprocess
from nba_highlights.pipeline import run_pipeline
from nba_highlights.config import Settings, Weights
from nba_highlights.transcribe.stub import StubTranscriber
from nba_highlights.action.stub import StubActionRecognizer
from nba_highlights.interfaces import Transcript, TranscriptSegment, Word
import numpy as np


def _fake_runner(cmd, **kw):
    if cmd and cmd[0] == "ffprobe":
        return subprocess.CompletedProcess(cmd, 0, stdout="40.0\n")
    return subprocess.CompletedProcess(cmd, 0)


def _transcript():
    # loud hype words land inside scene 1 (2.5-4.0)
    return Transcript(segments=[TranscriptSegment(
        start_s=2.6, end_s=3.9, text="what a dunk and one",
        words=[Word(text="what", start_s=2.6, end_s=2.8),
               Word(text="a", start_s=2.8, end_s=2.9),
               Word(text="dunk", start_s=3.0, end_s=3.3),
               Word(text="and", start_s=3.4, end_s=3.6),
               Word(text="one", start_s=3.6, end_s=3.9)])])


def _audio_loader(path):
    sr = 1000
    samples = np.zeros(sr * 40, dtype=float)
    samples[int(sr * 2.7):int(sr * 3.9)] = 3.0  # loud burst in scene 1
    return samples, sr


def _detector(path):
    return [(0.0, 2.0), (2.5, 4.0), (20.0, 22.0)]


def test_pipeline_detects_the_loud_hype_scene(tmp_path):
    settings = Settings(threshold=0.4, audio_gate=0.2, weights=Weights())
    action = StubActionRecognizer(scores={2: 0.9}, default=0.0)  # scene starting at 2.5
    report = run_pipeline(
        "/videos/game.mp4", settings,
        transcriber=StubTranscriber(_transcript()),
        action=action,
        workdir=str(tmp_path), generated_at="2026-07-13T00:00:00Z",
        title="Game 7",
        runner=_fake_runner, detector_fn=_detector, audio_loader=_audio_loader,
    )
    assert report.video.title == "Game 7"
    assert report.video.duration_s == 40.0
    assert len(report.highlights) == 1
    h = report.highlights[0]
    assert h.start_s == 2.5 and h.end_s == 4.0
    assert "dunk" in h.evidence.keywords and "and one" in h.evidence.keywords
    assert report.config["action_backend"] == "stub"


def test_pipeline_bad_config_raises(tmp_path):
    from nba_highlights.errors import ConfigError
    import pytest
    settings = Settings(weights=Weights(wA=0.9, wV=0.9))
    with pytest.raises(ConfigError):
        run_pipeline("/v.mp4", settings, StubTranscriber(), StubActionRecognizer(),
                     workdir=str(tmp_path), generated_at="t",
                     runner=_fake_runner, detector_fn=_detector, audio_loader=_audio_loader)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/nba_highlights/pipeline.py`**

```python
import logging
import subprocess

from nba_highlights import media, scenes
from nba_highlights.audio import loudness, prosody, keywords
from nba_highlights.config import Settings
from nba_highlights.fuse import percentile_normalize, audio_composite, fuse
from nba_highlights.interfaces import ActionRecognizer, Transcriber
from nba_highlights.merge import merge_shots
from nba_highlights.models import (
    AudioSignals, ActionSignals, ShotScore, Report, VideoInfo,
)
from nba_highlights.select import select_highlights

log = logging.getLogger(__name__)


def run_pipeline(
    source: str,
    settings: Settings,
    transcriber: Transcriber,
    action: ActionRecognizer,
    *,
    workdir: str,
    generated_at: str,
    title: str = "",
    downloader=media._default_downloader,
    runner=subprocess.run,
    detector_fn=scenes._default_detector,
    audio_loader=loudness.load_audio,
) -> Report:
    settings.check()
    video_path, audio_path = media.ingest(source, workdir,
                                          downloader=downloader, runner=runner)
    duration_s = media.probe_duration(video_path, runner=runner)
    shots = scenes.detect_scenes(video_path, duration_s, detector_fn=detector_fn)
    transcript = transcriber.transcribe(audio_path)
    samples, sr = audio_loader(audio_path)
    times, rms = loudness.rms_envelope(samples, sr)

    raw_loud: list[float] = []
    raw_pros: list[float] = []
    raw_kw: list[float] = []
    peak_times: list[float | None] = []
    matched: list[list[str]] = []
    texts: list[str] = []
    for shot in shots:
        try:
            peak, t_peak = loudness.shot_loudness(times, rms, shot.start_s, shot.end_s)
            pros = prosody.shot_prosody(samples, sr, shot.start_s, shot.end_s)
            kw_score, kw_matched = keywords.shot_keywords(
                transcript, shot.start_s, shot.end_s, settings.hype_lexicon)
        except Exception as e:  # never let one shot abort the run
            log.warning("audio scoring failed for scene %s: %s", shot.scene_id, e)
            peak, t_peak, pros, kw_score, kw_matched = 0.0, None, 0.0, 0.0, []
        raw_loud.append(peak)
        raw_pros.append(pros)
        raw_kw.append(kw_score)
        peak_times.append(t_peak)
        matched.append(kw_matched)
        texts.append(transcript.text_in(shot.start_s, shot.end_s))

    loud_n = percentile_normalize(raw_loud)
    pros_n = percentile_normalize(raw_pros)
    kw_n = percentile_normalize(raw_kw)

    shot_scores: list[ShotScore] = []
    for i, shot in enumerate(shots):
        comp = audio_composite(loud_n[i], pros_n[i], kw_n[i], settings.weights)
        act_score, act_label = 0.0, ""
        if settings.full_action or comp >= settings.audio_gate:
            try:
                res = action.score_clip(video_path, shot.start_s, shot.end_s)
                act_score, act_label = res.score, res.label
            except Exception as e:
                log.warning("action scoring failed for scene %s: %s", shot.scene_id, e)
        fused = fuse(comp, act_score, settings.weights)
        shot_scores.append(ShotScore(
            scene_id=shot.scene_id, start_s=shot.start_s, end_s=shot.end_s,
            audio=AudioSignals(loudness=loud_n[i], prosody=pros_n[i], keywords=kw_n[i]),
            action=ActionSignals(score=act_score, label=act_label),
            audio_composite=comp, fused=fused,
            matched_keywords=matched[i], peak_loudness_s=peak_times[i],
            transcript=texts[i]))

    merged = merge_shots(shot_scores, settings.threshold, settings.max_gap_s)
    highlights = select_highlights(merged, settings.threshold)

    config = {
        "weights": settings.weights.model_dump(),
        "threshold": settings.threshold,
        "audio_gate": settings.audio_gate,
        "max_gap_s": settings.max_gap_s,
        "action_backend": settings.action_backend,
        "asr_backend": settings.asr_backend,
        "asr_model": settings.asr_model,
        "full_action": settings.full_action,
    }
    return Report(
        video=VideoInfo(source=source, title=title, duration_s=duration_s),
        config=config, generated_at=generated_at, highlights=highlights)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -v`
Expected: 2 passed.

- [ ] **Step 5: Run the full suite so far**

Run: `.venv/bin/python -m pytest -q`
Expected: all passing.

- [ ] **Step 6: Commit**

```bash
git add src/nba_highlights/pipeline.py tests/test_pipeline.py
git commit -m "feat: end-to-end pipeline orchestration with two-pass action gate"
```

---

### Task 16: Eval harness (temporal IoU precision/recall)

**Files:**
- Create: `src/nba_highlights/eval.py`
- Test: `tests/test_eval.py`

**Interfaces:**
- Consumes: `nba_highlights.models.Report`.
- Produces:
  - `TruthSegment{start_s:float, end_s:float}`, `GroundTruth{highlights:list[TruthSegment]}` (pydantic; `GroundTruth` loadable from JSON).
  - `iou(a_start:float, a_end:float, b_start:float, b_end:float) -> float` (temporal intersection-over-union; disjoint → 0.0).
  - `EvalResult{precision:float, recall:float, f1:float, tp:int, fp:int, fn:int}`.
  - `score_report(report:Report, gt:GroundTruth, iou_threshold:float=0.5) -> EvalResult` — greedy one-to-one matching: each detected highlight matched to the unused truth segment with highest IoU ≥ threshold. Matched detection = TP; unmatched detection = FP; unmatched truth = FN.

- [ ] **Step 1: Write the failing test**

```python
from nba_highlights.eval import iou, score_report, GroundTruth, TruthSegment
from nba_highlights.models import (
    Report, VideoInfo, Highlight, SignalBreakdown, AudioSignals, ActionSignals, Evidence,
)


def _hl(hid, start, end):
    return Highlight(id=hid, start_s=start, end_s=end, score=0.9,
                     signals=SignalBreakdown(audio=AudioSignals(), action=ActionSignals()),
                     evidence=Evidence(), member_scenes=[hid])


def _report(spans):
    return Report(video=VideoInfo(source="u"), config={}, generated_at="t",
                  highlights=[_hl(i, s, e) for i, (s, e) in enumerate(spans)])


def test_iou_basic():
    assert iou(0.0, 10.0, 0.0, 10.0) == 1.0
    assert iou(0.0, 10.0, 5.0, 15.0) == 5.0 / 15.0
    assert iou(0.0, 5.0, 10.0, 15.0) == 0.0


def test_score_report_tp_fp_fn():
    report = _report([(0.0, 10.0), (100.0, 110.0)])   # 1 good, 1 spurious
    gt = GroundTruth(highlights=[TruthSegment(start_s=1.0, end_s=11.0),   # matches first
                                 TruthSegment(start_s=200.0, end_s=210.0)])  # missed
    r = score_report(report, gt, iou_threshold=0.5)
    assert r.tp == 1 and r.fp == 1 and r.fn == 1
    assert abs(r.precision - 0.5) < 1e-9
    assert abs(r.recall - 0.5) < 1e-9
    assert abs(r.f1 - 0.5) < 1e-9


def test_perfect_match():
    report = _report([(0.0, 10.0)])
    gt = GroundTruth(highlights=[TruthSegment(start_s=0.0, end_s=10.0)])
    r = score_report(report, gt)
    assert r.tp == 1 and r.fp == 0 and r.fn == 0 and r.f1 == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_eval.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/nba_highlights/eval.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_eval.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nba_highlights/eval.py tests/test_eval.py
git commit -m "feat: eval harness (temporal IoU precision/recall/F1)"
```

---

### Task 17: CLI (`analyze` + `eval`)

**Files:**
- Create: `src/nba_highlights/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `Settings`, `run_pipeline`, `score_report`, `GroundTruth`, `Report`, the stub + real backends.
- Produces:
  - typer `app` with two commands.
  - `analyze(source, out="report.json", workdir="./work", threshold=None, action_backend=None, asr_backend=None, full_action=False)` — builds `Settings` (env defaults, CLI overrides), constructs backends via `_build_backends(settings)`, runs `run_pipeline`, writes `report.model_dump_json(indent=2)`.
  - `eval(report, truth, out="eval.json")` — loads `Report` + `GroundTruth` from JSON, runs `score_report`, writes result JSON, echoes P/R/F1.
  - `_build_backends(settings) -> tuple[Transcriber, ActionRecognizer]` — maps `asr_backend`/`action_backend` strings to stub or real classes.

- [ ] **Step 1: Write the failing test**

```python
import json
from typer.testing import CliRunner
from nba_highlights.cli import app
from nba_highlights.models import (
    Report, VideoInfo, Highlight, SignalBreakdown, AudioSignals, ActionSignals, Evidence,
)

runner = CliRunner()


def test_analyze_stub_end_to_end(tmp_path, monkeypatch):
    # stub backends need no models; force a trivial file source through a fake ingest
    import nba_highlights.cli as climod

    def fake_run_pipeline(source, settings, transcriber, action, **kw):
        assert settings.action_backend == "stub"
        return Report(video=VideoInfo(source=source), config={}, generated_at="t",
                      highlights=[])

    monkeypatch.setattr(climod, "run_pipeline", fake_run_pipeline)
    out = tmp_path / "r.json"
    result = runner.invoke(app, ["analyze", "/v.mp4", "--out", str(out),
                                 "--workdir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text())["video"]["source"] == "/v.mp4"


def test_eval_command(tmp_path):
    report = Report(
        video=VideoInfo(source="u"), config={}, generated_at="t",
        highlights=[Highlight(id=0, start_s=0.0, end_s=10.0, score=0.9,
                    signals=SignalBreakdown(audio=AudioSignals(), action=ActionSignals()),
                    evidence=Evidence(), member_scenes=[0])])
    truth = {"highlights": [{"start_s": 0.0, "end_s": 10.0}]}
    rp = tmp_path / "report.json"
    tp = tmp_path / "truth.json"
    ep = tmp_path / "eval.json"
    rp.write_text(report.model_dump_json())
    tp.write_text(json.dumps(truth))
    result = runner.invoke(app, ["eval", "--report", str(rp), "--truth", str(tp),
                                 "--out", str(ep)])
    assert result.exit_code == 0, result.output
    assert json.loads(ep.read_text())["f1"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/nba_highlights/cli.py`**

```python
import os
import uuid
from datetime import datetime, timezone

import typer

from nba_highlights.config import Settings
from nba_highlights.eval import GroundTruth, score_report
from nba_highlights.models import Report
from nba_highlights.pipeline import run_pipeline

app = typer.Typer(help="NBA video highlight detection -> JSON report")


def _build_backends(settings: Settings):
    if settings.asr_backend == "faster-whisper":
        from nba_highlights.transcribe.faster_whisper_client import FasterWhisperTranscriber
        transcriber = FasterWhisperTranscriber(model_size=settings.asr_model)
    else:
        from nba_highlights.transcribe.stub import StubTranscriber
        transcriber = StubTranscriber()
    if settings.action_backend == "videomae":
        from nba_highlights.action.videomae_client import VideoMAEActionRecognizer
        action = VideoMAEActionRecognizer()
    else:
        from nba_highlights.action.stub import StubActionRecognizer
        action = StubActionRecognizer()
    return transcriber, action


@app.command()
def analyze(
    source: str = typer.Argument(..., help="YouTube URL, HLS .m3u8, or file path"),
    out: str = typer.Option("report.json", help="Output report path"),
    workdir: str = typer.Option("./work", help="Scratch dir for media"),
    threshold: float = typer.Option(None, help="Selection threshold override"),
    action_backend: str = typer.Option(None, help="stub | videomae"),
    asr_backend: str = typer.Option(None, help="stub | faster-whisper"),
    full_action: bool = typer.Option(False, help="Run action model on every shot"),
):
    settings = Settings.from_env()
    if threshold is not None:
        settings.threshold = threshold
    if action_backend is not None:
        settings.action_backend = action_backend
    if asr_backend is not None:
        settings.asr_backend = asr_backend
    settings.full_action = full_action
    settings.check()

    transcriber, action = _build_backends(settings)
    report = run_pipeline(
        source, settings, transcriber, action,
        workdir=workdir,
        generated_at=datetime.now(timezone.utc).isoformat(),
        title="", 
    )
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        f.write(report.model_dump_json(indent=2))
    typer.echo(f"wrote {out} ({len(report.highlights)} highlights)")


@app.command()
def eval(
    report: str = typer.Option(..., help="Path to a report.json"),
    truth: str = typer.Option(..., help="Path to a ground-truth JSON"),
    out: str = typer.Option("eval.json", help="Output eval path"),
):
    with open(report) as f:
        rep = Report.model_validate_json(f.read())
    with open(truth) as f:
        gt = GroundTruth.model_validate_json(f.read())
    result = score_report(rep, gt)
    with open(out, "w") as f:
        f.write(result.model_dump_json(indent=2))
    typer.echo(f"precision={result.precision:.3f} recall={result.recall:.3f} "
               f"f1={result.f1:.3f} (tp={result.tp} fp={result.fp} fn={result.fn})")
```

Note: the unused `uuid` import will be removed by ruff-fix in Step 5 if flagged; keep the file importing only what it uses. Remove the `import uuid` line before committing (it is not used).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_cli.py -v`
Expected: 2 passed.

- [ ] **Step 5: Lint and fix**

Run: `.venv/bin/ruff check --fix src tests`
Expected: no remaining errors (unused `uuid` import removed).

- [ ] **Step 6: Commit**

```bash
git add src/nba_highlights/cli.py tests/test_cli.py
git commit -m "feat: highlights CLI (analyze + eval)"
```

---

### Task 18: Integration test (slow) + README

**Files:**
- Create: `tests/test_integration_slow.py`
- Create: `README.md`

**Interfaces:**
- Consumes: everything.
- Produces: one `slow`-marked integration test that runs the pipeline on a tiny synthetic local MP4 with **real** scene detection but stub ML backends (no model download, but exercises ffmpeg + scenedetect if installed); skips cleanly when `media`/`audio` extras are absent. Plus a README documenting install, run, config, and eval.

- [ ] **Step 1: Write the integration test**

```python
import shutil
import subprocess
import pytest

pytest.importorskip("scenedetect")
pytest.importorskip("soundfile")

pytestmark = pytest.mark.slow


def _make_clip(path):
    # 6s test video with a scene change at 3s (color flip) + a tone, via ffmpeg
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "lavfi", "-i", "color=c=red:s=320x240:d=3",
         "-f", "lavfi", "-i", "color=c=blue:s=320x240:d=3",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
         "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
         "-map", "[v]", "-map", "2:a", "-shortest", path],
        check=True)


def test_pipeline_on_real_clip(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    from nba_highlights.pipeline import run_pipeline
    from nba_highlights.config import Settings
    from nba_highlights.transcribe.stub import StubTranscriber
    from nba_highlights.action.stub import StubActionRecognizer

    clip = str(tmp_path / "clip.mp4")
    _make_clip(clip)
    settings = Settings(threshold=0.0, audio_gate=0.0)  # accept everything
    report = run_pipeline(
        clip, settings, StubTranscriber(), StubActionRecognizer(default=0.5),
        workdir=str(tmp_path / "work"), generated_at="2026-07-13T00:00:00Z")
    assert report.video.duration_s > 0
    assert isinstance(report.highlights, list)
```

- [ ] **Step 2: Run the slow test (only if extras installed)**

Run: `.venv/bin/pip install -e '.[dev,media,audio]' && .venv/bin/python -m pytest -m slow -v`
Expected: passes, or skips if ffmpeg/extras missing. (Default `pytest -q` still deselects it.)

- [ ] **Step 3: Write `README.md`**

````markdown
# nba-highlights

Detect **play-level highlights** in an NBA video and emit a **JSON report** with
timestamps, a fused audio+action score, and per-signal evidence.

Input: a YouTube URL (full game), HLS `.m3u8`, or a local file.
Output: JSON — each highlight has `start_s`/`end_s`, `score`, `signals`
(audio: loudness/prosody/keywords; action: score/label), and `evidence`.

## How it works

```
video ─▶ ingest (yt-dlp/ffmpeg) ─▶ scene detect (PySceneDetect)
      ─▶ per shot: audio (loudness + prosody + ASR keyword spotting)
      ─▶ two-pass: action recognition (VideoMAE) on audio-gated candidate shots
      ─▶ weighted fusion ─▶ merge adjacent high shots into plays ─▶ threshold ─▶ JSON
```

Scores are percentile-normalized per game, so the threshold is game-relative.
The **two-pass gate** runs the expensive video model only on audio-loud
candidate shots (override with `--full-action`).

## Install

```bash
python3 -m venv .venv && .venv/bin/pip install -U pip
.venv/bin/pip install -e '.[dev]'                      # tests only (stub backends)
.venv/bin/pip install -e '.[dev,media,audio,asr,action]'   # real run
```

## Run

```bash
# stub backends (deterministic, no models):
.venv/bin/highlights analyze /path/to/game.mp4 --out report.json

# real models:
.venv/bin/highlights analyze "https://www.youtube.com/watch?v=<id>" \
  --out report.json --workdir ./work \
  --action-backend videomae --asr-backend faster-whisper \
  [--threshold 0.5] [--full-action]
```

## Evaluate

```bash
.venv/bin/highlights eval --report report.json --truth truth.json --out eval.json
# truth.json: {"highlights": [{"start_s": 1423.5, "end_s": 1431.2}, ...]}
```
Reports precision/recall/F1 by temporal IoU ≥ 0.5.

## Config (env vars)

| Var | Default | Meaning |
|---|---|---|
| `ACTION_BACKEND` | `stub` | `stub` \| `videomae` |
| `ASR_BACKEND` | `stub` | `stub` \| `faster-whisper` |
| `ASR_MODEL` | `small` | faster-whisper model size |
| `THRESHOLD` | `0.5` | selection threshold |
| `AUDIO_GATE` | `0.35` | audio pre-gate for the action model |
| `MAX_GAP_S` | `3.0` | max gap to merge adjacent shots |

## Testing

```bash
.venv/bin/python -m pytest -q          # fast unit tests (stub backends, no models)
.venv/bin/python -m pytest -m slow -v  # integration on a real clip (needs ffmpeg + extras)
.venv/bin/ruff check src tests
```

## Design

See `docs/superpowers/specs/2026-07-13-nba-highlights-design.md` and
`docs/superpowers/plans/2026-07-13-nba-highlights.md`.
````

- [ ] **Step 4: Full suite + lint**

Run:
```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests
```
Expected: all unit tests pass; lint clean.

- [ ] **Step 5: Commit**

```bash
git add tests/test_integration_slow.py README.md
git commit -m "test: slow integration on real clip; docs: README"
```

---

## Self-Review

**1. Spec coverage:**
- §1 goal (YouTube full-game → JSON) → Tasks 5, 15, 17. ✔
- §2 pipeline stages (ingest/scene/audio/action/fuse/merge/select) → Tasks 5,6,8-14. ✔
- §2 two-pass audio gate + `--full-action` → Task 15 (gate), Task 4 (`audio_gate`/`full_action`), Task 17 (flag). ✔
- §3 three audio signals + percentile normalization + weighted fusion → Tasks 8,9,10,12,15. ✔
- §4 output JSON schema → Task 2 models, Task 15 assembly. ✔
- §5 selection (weighted score + threshold) → Tasks 13,14,15. ✔
- §6 error handling (IngestError, per-shot wrap, ASR graceful, config fail-loud) → Tasks 1,5,15,4. ✔
- §7 testing + eval (stub-backend unit, pure-function, slow integration, IoU eval) → all tasks + 16,18. ✔
- §8 repo layout + extras → Task 1 pyproject + per-module tasks. ✔
- §9 CLI (analyze + eval) → Task 17. ✔
- §10 out of scope — respected (no clip rendering, no async, no Triton). ✔

**2. Placeholder scan:** No "TBD"/"handle edge cases"/"similar to". Every code step shows full code. Task 17 explicitly flags removing the unused `uuid` import. ✔

**3. Type consistency:** `Transcript.text_in` (Task 3) used by keywords (10) + pipeline (15). `ShotScore` fields (Task 2) written in pipeline (15), read in merge (13). `Highlight`/`Evidence`/`SignalBreakdown` (Task 2) built in merge (13), filtered in select (14). `Weights` (Task 4) consumed by fuse (12) + pipeline (15). `ActionResult` (Task 3) returned by action backends (11), consumed in pipeline (15). `score_from_probs`/`StubActionRecognizer` names match between Task 11 def and Task 15/17 use. `run_pipeline` signature (Task 15) matches CLI call (Task 17) and test monkeypatch. ✔
