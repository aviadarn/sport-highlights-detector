# M2 Learned Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add labeled evaluation + a learned highlight classifier that plugs in behind a swappable fusion seam, defaulting to the existing M1 weighted logic.

**Architecture:** Extract the per-shot fusion decision into a `HighlightScorer` seam. `WeightedScorer` reproduces M1's `audio_composite`+`fuse` exactly (default). `LearnedScorer` loads a persisted scikit-learn model. The pipeline builds a frozen per-shot `ShotFeatures` vector, scores each shot via the injected scorer, and can emit features for training; the two-pass action gate stays gated on `audio_composite` (cost), independent of the scorer. An offline training script joins emitted features to human-labeled truth JSON (by temporal IoU, reusing `eval.iou`) and fits a classifier with leave-one-game-out CV.

**Tech Stack:** Python 3.11+, pydantic v2, typer, numpy. New optional `train` extra: scikit-learn + joblib (lazy-imported, never in the default test suite).

## Global Constraints

- Python `>=3.11`. Package `nba_highlights`, source under `src/nba_highlights/`.
- scikit-learn and joblib live ONLY in a new `train` optional extra and are **lazy-imported inside functions**. `pip install -e '.[dev]'` alone must run the entire test suite with no sklearn/joblib import. Tests that need them use `pytest.importorskip`, or inject a fake model so they don't need them at all.
- `WeightedScorer` is the default fusion and MUST reproduce M1 fusion exactly: `fuse(audio_composite(loudness_n, prosody_n, keywords_n, weights), action_score, weights)`. The existing `tests/test_pipeline.py` must keep passing unchanged (regression guarantee).
- The per-shot feature vector column order is `FEATURE_NAMES` and `to_vector` MUST follow it exactly — training and inference share this ordering.
- Truth JSON format is unchanged: `eval.GroundTruth` = `{highlights: [{start_s, end_s}]}`. Label join uses `eval.iou` with threshold 0.5 (a shot is positive iff it overlaps any truth segment with IoU ≥ 0.5).
- The two-pass action gate remains gated on `audio_composite >= settings.audio_gate` (or `full_action`) — a COST decision, independent of which scorer produces the final score.
- ruff `line-length = 100`, `target-version = "py311"`. TDD: failing test → run → implement → pass → commit.

---

### Task 1: ShotFeatures model + feature vector

**Files:**
- Create: `src/nba_highlights/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `FEATURE_NAMES: list[str]` = `["loudness_n","prosody_n","keywords_n","keyword_count","action_score","shot_duration_s","rel_position"]`
  - `ShotFeatures` (pydantic): `scene_id:int, start_s:float, end_s:float`, plus the seven feature fields above (all `float`, default `0.0`).
  - `to_vector(feats:ShotFeatures) -> list[float]` returning the seven feature values in `FEATURE_NAMES` order.

- [ ] **Step 1: Write the failing test**

```python
from nba_highlights.features import ShotFeatures, FEATURE_NAMES, to_vector


def test_feature_names_frozen_order():
    assert FEATURE_NAMES == ["loudness_n", "prosody_n", "keywords_n",
                             "keyword_count", "action_score", "shot_duration_s",
                             "rel_position"]


def test_to_vector_follows_feature_names():
    f = ShotFeatures(scene_id=1, start_s=0.0, end_s=2.0, loudness_n=0.9,
                     prosody_n=0.8, keywords_n=0.6, keyword_count=2.0,
                     action_score=0.7, shot_duration_s=2.0, rel_position=0.5)
    assert to_vector(f) == [0.9, 0.8, 0.6, 2.0, 0.7, 2.0, 0.5]


def test_defaults_zero():
    f = ShotFeatures(scene_id=0, start_s=0.0, end_s=1.0)
    assert to_vector(f) == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nba_highlights.features'`.

- [ ] **Step 3: Write `src/nba_highlights/features.py`**

```python
from pydantic import BaseModel

FEATURE_NAMES = [
    "loudness_n", "prosody_n", "keywords_n", "keyword_count",
    "action_score", "shot_duration_s", "rel_position",
]


class ShotFeatures(BaseModel):
    scene_id: int
    start_s: float
    end_s: float
    loudness_n: float = 0.0
    prosody_n: float = 0.0
    keywords_n: float = 0.0
    keyword_count: float = 0.0
    action_score: float = 0.0
    shot_duration_s: float = 0.0
    rel_position: float = 0.0


def to_vector(feats: ShotFeatures) -> list[float]:
    return [getattr(feats, name) for name in FEATURE_NAMES]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_features.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nba_highlights/features.py tests/test_features.py
git commit -m "feat: ShotFeatures model + frozen feature vector"
```

---

### Task 2: Scorer seam — protocol + WeightedScorer + StubScorer

**Files:**
- Create: `src/nba_highlights/scoring/__init__.py`
- Create: `src/nba_highlights/scoring/base.py`
- Create: `src/nba_highlights/scoring/weighted.py`
- Create: `src/nba_highlights/scoring/stub.py`
- Test: `tests/test_scoring.py`

**Interfaces:**
- Consumes: `nba_highlights.features.ShotFeatures`, `nba_highlights.config.Weights`, `nba_highlights.fuse.{audio_composite, fuse}`.
- Produces:
  - `HighlightScorer` protocol (`scoring/base.py`): `score(self, feats: ShotFeatures) -> float`.
  - `WeightedScorer(weights: Weights)` (`scoring/weighted.py`): `.score(feats)` = `fuse(audio_composite(feats.loudness_n, feats.prosody_n, feats.keywords_n, weights), feats.action_score, weights)`.
  - `StubScorer(scores: dict[int, float] | None = None, default: float = 0.0)` (`scoring/stub.py`): `.score(feats)` = `scores.get(feats.scene_id, default)`.

- [ ] **Step 1: Write the failing test**

```python
from nba_highlights.features import ShotFeatures
from nba_highlights.config import Weights
from nba_highlights.fuse import audio_composite, fuse
from nba_highlights.scoring.weighted import WeightedScorer
from nba_highlights.scoring.stub import StubScorer


def test_weighted_scorer_matches_m1_fusion():
    w = Weights()
    f = ShotFeatures(scene_id=0, start_s=0.0, end_s=2.0,
                     loudness_n=0.9, prosody_n=0.8, keywords_n=0.6, action_score=0.7)
    expected = fuse(audio_composite(0.9, 0.8, 0.6, w), 0.7, w)
    assert WeightedScorer(w).score(f) == expected


def test_stub_scorer_keyed_by_scene_id():
    s = StubScorer(scores={5: 0.9}, default=0.1)
    hot = ShotFeatures(scene_id=5, start_s=0.0, end_s=1.0)
    cold = ShotFeatures(scene_id=6, start_s=0.0, end_s=1.0)
    assert s.score(hot) == 0.9
    assert s.score(cold) == 0.1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_scoring.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the files**

`src/nba_highlights/scoring/__init__.py`:
```python
```

`src/nba_highlights/scoring/base.py`:
```python
from typing import Protocol
from nba_highlights.features import ShotFeatures


class HighlightScorer(Protocol):
    def score(self, feats: ShotFeatures) -> float: ...
```

`src/nba_highlights/scoring/weighted.py`:
```python
from nba_highlights.config import Weights
from nba_highlights.features import ShotFeatures
from nba_highlights.fuse import audio_composite, fuse


class WeightedScorer:
    def __init__(self, weights: Weights) -> None:
        self._w = weights

    def score(self, feats: ShotFeatures) -> float:
        comp = audio_composite(feats.loudness_n, feats.prosody_n,
                               feats.keywords_n, self._w)
        return fuse(comp, feats.action_score, self._w)
```

`src/nba_highlights/scoring/stub.py`:
```python
from nba_highlights.features import ShotFeatures


class StubScorer:
    def __init__(self, scores: dict[int, float] | None = None,
                 default: float = 0.0) -> None:
        self._scores = scores or {}
        self._default = default

    def score(self, feats: ShotFeatures) -> float:
        return self._scores.get(feats.scene_id, self._default)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_scoring.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nba_highlights/scoring/__init__.py src/nba_highlights/scoring/base.py src/nba_highlights/scoring/weighted.py src/nba_highlights/scoring/stub.py tests/test_scoring.py
git commit -m "feat: HighlightScorer seam with WeightedScorer + StubScorer"
```

---

### Task 3: LearnedScorer + scorer factory

**Files:**
- Create: `src/nba_highlights/scoring/learned.py`
- Create: `src/nba_highlights/scoring/factory.py`
- Test: `tests/test_learned_scorer.py`

**Interfaces:**
- Consumes: `nba_highlights.features.{ShotFeatures, to_vector}`, `nba_highlights.config.Weights`, `nba_highlights.scoring.weighted.WeightedScorer`.
- Produces:
  - `LearnedScorer(model_path: str | None = None, model=None)` (`scoring/learned.py`): `.score(feats)` returns `float(model.predict_proba([to_vector(feats)])[0][1])`. If `model` is None it lazy-loads via `joblib.load(model_path)`; an injected `model` skips joblib entirely (for tests).
  - `load_scorer(fusion: str, model_path: str, weights: Weights) -> HighlightScorer` (`scoring/factory.py`): returns `LearnedScorer(model_path)` iff `fusion == "learned"` and `model_path` is set and exists on disk; otherwise `WeightedScorer(weights)`.

- [ ] **Step 1: Write the failing test**

```python
from nba_highlights.features import ShotFeatures
from nba_highlights.config import Weights
from nba_highlights.scoring.learned import LearnedScorer
from nba_highlights.scoring.factory import load_scorer
from nba_highlights.scoring.weighted import WeightedScorer


class _FakeModel:
    def predict_proba(self, X):
        assert len(X) == 1 and len(X[0]) == 7  # one row, seven features
        return [[0.3, 0.7]]


def test_learned_scorer_returns_positive_class_proba():
    f = ShotFeatures(scene_id=0, start_s=0.0, end_s=1.0, action_score=0.9)
    assert LearnedScorer(model=_FakeModel()).score(f) == 0.7


def test_load_scorer_falls_back_to_weighted_when_model_missing(tmp_path):
    s = load_scorer("learned", str(tmp_path / "nope.joblib"), Weights())
    assert isinstance(s, WeightedScorer)


def test_load_scorer_weighted_when_fusion_weighted():
    assert isinstance(load_scorer("weighted", "", Weights()), WeightedScorer)


def test_load_scorer_learned_when_model_exists(tmp_path):
    p = tmp_path / "m.joblib"
    p.write_bytes(b"x")  # existence is enough; loading is lazy
    s = load_scorer("learned", str(p), Weights())
    assert isinstance(s, LearnedScorer)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_learned_scorer.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the files**

`src/nba_highlights/scoring/learned.py`:
```python
from nba_highlights.features import ShotFeatures, to_vector


class LearnedScorer:
    def __init__(self, model_path: str | None = None, model=None) -> None:
        self._model_path = model_path
        self._model = model

    def _load(self):
        if self._model is None:
            import joblib
            self._model = joblib.load(self._model_path)
        return self._model

    def score(self, feats: ShotFeatures) -> float:
        model = self._load()
        return float(model.predict_proba([to_vector(feats)])[0][1])
```

`src/nba_highlights/scoring/factory.py`:
```python
import os
from nba_highlights.config import Weights
from nba_highlights.scoring.weighted import WeightedScorer


def load_scorer(fusion: str, model_path: str, weights: Weights):
    if fusion == "learned" and model_path and os.path.exists(model_path):
        from nba_highlights.scoring.learned import LearnedScorer
        return LearnedScorer(model_path)
    return WeightedScorer(weights)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_learned_scorer.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nba_highlights/scoring/learned.py src/nba_highlights/scoring/factory.py tests/test_learned_scorer.py
git commit -m "feat: LearnedScorer (lazy joblib) + load_scorer fallback factory"
```

---

### Task 4: Config — fusion backend fields

**Files:**
- Modify: `src/nba_highlights/config.py`
- Test: `tests/test_config.py` (add cases)

**Interfaces:**
- Consumes: `nba_highlights.errors.ConfigError`.
- Produces: `Settings` gains `fusion: str = "weighted"` and `fusion_model: str = ""`. `FUSION_BACKENDS = {"weighted", "learned"}`. `check()` raises `ConfigError` on an unknown `fusion`. `from_env()` reads `FUSION` and `FUSION_MODEL`.

- [ ] **Step 1: Write the failing test (append to `tests/test_config.py`)**

```python
def test_fusion_defaults_and_validation():
    from nba_highlights.config import Settings
    from nba_highlights.errors import ConfigError
    import pytest
    s = Settings()
    assert s.fusion == "weighted" and s.fusion_model == ""
    with pytest.raises(ConfigError):
        Settings(fusion="bogus").check()


def test_from_env_reads_fusion(monkeypatch):
    from nba_highlights.config import Settings
    monkeypatch.setenv("FUSION", "learned")
    monkeypatch.setenv("FUSION_MODEL", "/tmp/m.joblib")
    s = Settings.from_env()
    assert s.fusion == "learned" and s.fusion_model == "/tmp/m.joblib"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_config.py -k fusion -v`
Expected: FAIL (`fusion` attribute / env handling absent; the `learned` fusion currently rejected or unset).

- [ ] **Step 3: Edit `src/nba_highlights/config.py`**

Add near the other backend sets:
```python
FUSION_BACKENDS = {"weighted", "learned"}
```

Add two fields to `Settings` (beside `full_action`):
```python
    fusion: str = "weighted"
    fusion_model: str = ""
```

In `check()`, add after the asr_backend check:
```python
        if self.fusion not in FUSION_BACKENDS:
            raise ConfigError(f"unknown fusion: {self.fusion!r}")
```

In `from_env()`, add before `s.check()`:
```python
        s.fusion = os.environ.get("FUSION", s.fusion)
        s.fusion_model = os.environ.get("FUSION_MODEL", s.fusion_model)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: all config tests pass (new + existing).

- [ ] **Step 5: Commit**

```bash
git add src/nba_highlights/config.py tests/test_config.py
git commit -m "feat: fusion + fusion_model settings with validation and env override"
```

---

### Task 5: Pipeline integration — scorer seam + feature emission

**Files:**
- Modify: `src/nba_highlights/pipeline.py`
- Test: `tests/test_pipeline_features.py`

**Interfaces:**
- Consumes: `nba_highlights.features.ShotFeatures`, `nba_highlights.scoring.weighted.WeightedScorer`, `nba_highlights.scoring.base.HighlightScorer`.
- Produces: `run_pipeline` gains two keyword params: `scorer: HighlightScorer | None = None` (defaults to `WeightedScorer(settings.weights)` when None) and `emit_features_path: str | None = None`. Per shot the pipeline builds a `ShotFeatures` (with `keyword_count=float(len(matched[i]))`, `shot_duration_s=max(0.0, end-start)`, `rel_position = start_s/duration_s if duration_s>0 else 0.0`), computes `fused = scorer.score(feats)`, and — if `emit_features_path` is set — writes one `feats.model_dump_json()` per line. `audio_composite` is still computed and still gates the action model. The `config` dict gains `"fusion"`.

- [ ] **Step 1: Write the failing test**

```python
import subprocess
import numpy as np
from nba_highlights.pipeline import run_pipeline
from nba_highlights.config import Settings, Weights
from nba_highlights.transcribe.stub import StubTranscriber
from nba_highlights.action.stub import StubActionRecognizer
from nba_highlights.scoring.stub import StubScorer
from nba_highlights.features import ShotFeatures


def _runner(cmd, **kw):
    if cmd and cmd[0] == "ffprobe":
        return subprocess.CompletedProcess(cmd, 0, stdout="40.0\n")
    return subprocess.CompletedProcess(cmd, 0)


def _audio_loader(path):
    sr = 1000
    samples = np.zeros(sr * 40, dtype=float)
    samples[int(sr * 2.7):int(sr * 3.9)] = 3.0
    return samples, sr


def _detector(path):
    return [(0.0, 2.0), (2.5, 4.0), (20.0, 22.0)]


def test_emit_features_writes_one_row_per_shot(tmp_path):
    feats_path = tmp_path / "feats.jsonl"
    run_pipeline(
        "/v.mp4", Settings(threshold=0.4, audio_gate=0.2), StubTranscriber(),
        StubActionRecognizer(scores={2: 0.9}),
        workdir=str(tmp_path), generated_at="t",
        runner=_runner, detector_fn=_detector, audio_loader=_audio_loader,
        emit_features_path=str(feats_path))
    rows = [ShotFeatures.model_validate_json(ln)
            for ln in feats_path.read_text().splitlines() if ln.strip()]
    assert len(rows) == 3                      # one per detected shot
    assert rows[1].scene_id == 1
    assert 0.0 <= rows[1].rel_position <= 1.0
    assert rows[1].shot_duration_s == 1.5      # 4.0 - 2.5


def test_injected_scorer_drives_selection(tmp_path):
    # StubScorer marks only scene 2 hot; it must be the sole highlight.
    report = run_pipeline(
        "/v.mp4", Settings(threshold=0.5, audio_gate=0.0), StubTranscriber(),
        StubActionRecognizer(),
        workdir=str(tmp_path), generated_at="t",
        runner=_runner, detector_fn=_detector, audio_loader=_audio_loader,
        scorer=StubScorer(scores={2: 0.95}, default=0.0))
    assert len(report.highlights) == 1
    assert report.highlights[0].member_scenes == [2]
    assert report.config["fusion"] == "weighted"


def test_weighted_default_is_unchanged(tmp_path):
    # Regression: default (no scorer) still finds the loud+hype scene.
    report = run_pipeline(
        "/v.mp4", Settings(threshold=0.4, audio_gate=0.2), StubTranscriber(),
        StubActionRecognizer(scores={2: 0.9}),
        workdir=str(tmp_path), generated_at="t",
        runner=_runner, detector_fn=_detector, audio_loader=_audio_loader)
    assert len(report.highlights) == 1
```

Note: `test_injected_scorer_drives_selection` asserts `config["fusion"] == "weighted"` because the `config` dict records `settings.fusion` (the injected scorer is orthogonal to the recorded setting). The scene-2 stub score 0.95 ≥ threshold 0.5 makes it the sole highlight; scenes 0/1 score 0.0.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pipeline_features.py -v`
Expected: FAIL (`run_pipeline() got an unexpected keyword argument 'emit_features_path'`).

- [ ] **Step 3: Edit `src/nba_highlights/pipeline.py`**

Add imports near the top:
```python
from nba_highlights.features import ShotFeatures
from nba_highlights.scoring.weighted import WeightedScorer
```

Add two params to `run_pipeline`'s keyword-only block (after `audio_loader=...`):
```python
    scorer=None,
    emit_features_path: str | None = None,
```

Immediately after `settings.check()`:
```python
    if scorer is None:
        scorer = WeightedScorer(settings.weights)
```

Replace the second per-shot loop (the `for i, shot in enumerate(shots):` block that builds `shot_scores`) with:
```python
    feats_list: list[ShotFeatures] = []
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
        feats = ShotFeatures(
            scene_id=shot.scene_id, start_s=shot.start_s, end_s=shot.end_s,
            loudness_n=loud_n[i], prosody_n=pros_n[i], keywords_n=kw_n[i],
            keyword_count=float(len(matched[i])), action_score=act_score,
            shot_duration_s=max(0.0, shot.end_s - shot.start_s),
            rel_position=(shot.start_s / duration_s) if duration_s > 0 else 0.0)
        feats_list.append(feats)
        fused = scorer.score(feats)
        shot_scores.append(ShotScore(
            scene_id=shot.scene_id, start_s=shot.start_s, end_s=shot.end_s,
            audio=AudioSignals(loudness=loud_n[i], prosody=pros_n[i], keywords=kw_n[i]),
            action=ActionSignals(score=act_score, label=act_label),
            audio_composite=comp, fused=fused,
            matched_keywords=matched[i], peak_loudness_s=peak_times[i],
            transcript=texts[i]))

    if emit_features_path:
        with open(emit_features_path, "w") as fh:
            for f in feats_list:
                fh.write(f.model_dump_json() + "\n")
```

In the `config` dict, add:
```python
        "fusion": settings.fusion,
```

- [ ] **Step 4: Run tests to verify they pass (incl. regression)**

Run:
```bash
.venv/bin/python -m pytest tests/test_pipeline_features.py tests/test_pipeline.py -v
```
Expected: new feature tests pass AND the original `tests/test_pipeline.py` still passes (WeightedScorer default reproduces M1).

- [ ] **Step 5: Commit**

```bash
git add src/nba_highlights/pipeline.py tests/test_pipeline_features.py
git commit -m "feat: pipeline uses HighlightScorer seam + emits ShotFeatures"
```

---

### Task 6: Training dataset — feature↔label join

**Files:**
- Create: `src/nba_highlights/training/__init__.py`
- Create: `src/nba_highlights/training/dataset.py`
- Test: `tests/test_training_dataset.py`

**Interfaces:**
- Consumes: `nba_highlights.features.{ShotFeatures, to_vector}`, `nba_highlights.eval.{GroundTruth, iou}`.
- Produces:
  - `label_shots(features: list[ShotFeatures], gt: GroundTruth, iou_threshold: float = 0.5) -> list[tuple[ShotFeatures, int]]` — label 1 iff the shot overlaps any truth segment with IoU ≥ threshold.
  - `load_features(path: str) -> list[ShotFeatures]` — parse a JSONL feature file.
  - `build_table(pairs: list[tuple[str, str]], iou_threshold: float = 0.5) -> tuple[list[list[float]], list[int], list[int]]` — for each `(features_jsonl, truth_json)` game, join and accumulate `(X rows, y labels, group index per row)`.

- [ ] **Step 1: Write the failing test**

```python
import json
from nba_highlights.features import ShotFeatures
from nba_highlights.eval import GroundTruth, TruthSegment
from nba_highlights.training.dataset import label_shots, load_features, build_table


def test_label_shots_by_iou():
    feats = [ShotFeatures(scene_id=0, start_s=0.0, end_s=10.0),   # overlaps truth
             ShotFeatures(scene_id=1, start_s=100.0, end_s=110.0)]  # no truth
    gt = GroundTruth(highlights=[TruthSegment(start_s=1.0, end_s=11.0)])
    labeled = label_shots(feats, gt, iou_threshold=0.5)
    assert [lab for _, lab in labeled] == [1, 0]


def test_load_features_and_build_table(tmp_path):
    fpath = tmp_path / "g1.jsonl"
    fpath.write_text("\n".join(
        ShotFeatures(scene_id=i, start_s=float(i * 10), end_s=float(i * 10 + 10),
                     action_score=0.9 if i == 0 else 0.0).model_dump_json()
        for i in range(2)))
    tpath = tmp_path / "g1.truth.json"
    tpath.write_text(json.dumps({"highlights": [{"start_s": 0.0, "end_s": 10.0}]}))

    assert len(load_features(str(fpath))) == 2
    X, y, groups = build_table([(str(fpath), str(tpath))])
    assert len(X) == 2 and len(X[0]) == 7
    assert y == [1, 0]
    assert groups == [0, 0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_training_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the files**

`src/nba_highlights/training/__init__.py`:
```python
```

`src/nba_highlights/training/dataset.py`:
```python
from nba_highlights.eval import GroundTruth, iou
from nba_highlights.features import ShotFeatures, to_vector


def label_shots(features: list[ShotFeatures], gt: GroundTruth,
                iou_threshold: float = 0.5) -> list[tuple[ShotFeatures, int]]:
    out: list[tuple[ShotFeatures, int]] = []
    for f in features:
        positive = any(
            iou(f.start_s, f.end_s, t.start_s, t.end_s) >= iou_threshold
            for t in gt.highlights)
        out.append((f, 1 if positive else 0))
    return out


def load_features(path: str) -> list[ShotFeatures]:
    with open(path) as fh:
        return [ShotFeatures.model_validate_json(line)
                for line in fh if line.strip()]


def build_table(pairs: list[tuple[str, str]], iou_threshold: float = 0.5):
    X: list[list[float]] = []
    y: list[int] = []
    groups: list[int] = []
    for group_index, (feat_path, truth_path) in enumerate(pairs):
        feats = load_features(feat_path)
        with open(truth_path) as fh:
            gt = GroundTruth.model_validate_json(fh.read())
        for f, label in label_shots(feats, gt, iou_threshold):
            X.append(to_vector(f))
            y.append(label)
            groups.append(group_index)
    return X, y, groups
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_training_dataset.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nba_highlights/training/__init__.py src/nba_highlights/training/dataset.py tests/test_training_dataset.py
git commit -m "feat: training dataset builder (IoU feature-label join)"
```

---

### Task 7: Training function + script + `train` extra

**Files:**
- Modify: `pyproject.toml` (add `train` extra)
- Create: `src/nba_highlights/training/train.py`
- Create: `scripts/train_fusion.py`
- Test: `tests/test_train_fusion.py`

**Interfaces:**
- Consumes: `nba_highlights.training.dataset.build_table`.
- Produces:
  - `train_model(pairs: list[tuple[str, str]], out_path: str, iou_threshold: float = 0.5) -> dict` (`training/train.py`): builds the table, fits an L2 `LogisticRegression` on all rows, writes the fitted model to `out_path` via joblib, and returns a metrics dict `{"n": int, "positives": int, "cv_f1": float | None}`. `cv_f1` is the mean leave-one-game-out F1 when there are ≥2 groups and both classes are present, else `None`. sklearn + joblib are imported INSIDE the function.
  - `scripts/train_fusion.py`: CLI wrapper — `python scripts/train_fusion.py --pair feats1.jsonl truth1.json --pair feats2.jsonl truth2.json --out model.joblib`, prints the metrics dict.

- [ ] **Step 1: Add the `train` extra to `pyproject.toml`**

Under `[project.optional-dependencies]`, add:
```toml
train = ["scikit-learn>=1.3", "joblib>=1.3"]
```

- [ ] **Step 2: Write the failing test**

```python
import json
import pytest

pytest.importorskip("sklearn")

from nba_highlights.features import ShotFeatures
from nba_highlights.training.train import train_model


def _game(tmp_path, name, hot_scene):
    fp = tmp_path / f"{name}.jsonl"
    fp.write_text("\n".join(
        ShotFeatures(scene_id=i, start_s=float(i * 10), end_s=float(i * 10 + 10),
                     loudness_n=1.0 if i == hot_scene else 0.0,
                     action_score=1.0 if i == hot_scene else 0.0).model_dump_json()
        for i in range(4)))
    tp = tmp_path / f"{name}.truth.json"
    tp.write_text(json.dumps(
        {"highlights": [{"start_s": hot_scene * 10.0, "end_s": hot_scene * 10 + 10.0}]}))
    return (str(fp), str(tp))


def test_train_model_fits_and_persists(tmp_path):
    pairs = [_game(tmp_path, "g1", 1), _game(tmp_path, "g2", 2)]
    out = tmp_path / "model.joblib"
    metrics = train_model(pairs, str(out))
    assert out.exists()
    assert metrics["n"] == 8 and metrics["positives"] == 2
    assert metrics["cv_f1"] is not None

    # The persisted model loads and scores the separable signal correctly.
    import joblib
    from nba_highlights.features import to_vector
    model = joblib.load(str(out))
    hot = to_vector(ShotFeatures(scene_id=0, start_s=0, end_s=10,
                                 loudness_n=1.0, action_score=1.0))
    cold = to_vector(ShotFeatures(scene_id=0, start_s=0, end_s=10))
    assert model.predict_proba([hot])[0][1] > model.predict_proba([cold])[0][1]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pip install -e '.[dev,train]' && .venv/bin/python -m pytest tests/test_train_fusion.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nba_highlights.training.train'`.

- [ ] **Step 4: Write `src/nba_highlights/training/train.py`**

```python
from nba_highlights.training.dataset import build_table


def train_model(pairs: list[tuple[str, str]], out_path: str,
                iou_threshold: float = 0.5) -> dict:
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import LeaveOneGroupOut

    X, y, groups = build_table(pairs, iou_threshold)
    cv_f1 = None
    n_groups = len(set(groups))
    if n_groups >= 2 and len(set(y)) == 2:
        scores: list[float] = []
        logo = LeaveOneGroupOut()
        for train_idx, test_idx in logo.split(X, y, groups):
            y_train = [y[i] for i in train_idx]
            if len(set(y_train)) < 2:  # fold's train split is single-class
                continue
            clf = LogisticRegression(max_iter=1000, C=1.0)
            clf.fit([X[i] for i in train_idx], y_train)
            preds = clf.predict([X[i] for i in test_idx])
            scores.append(f1_score([y[i] for i in test_idx], preds,
                                   zero_division=0))
        cv_f1 = sum(scores) / len(scores) if scores else None

    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X, y)
    joblib.dump(model, out_path)
    return {"n": len(y), "positives": int(sum(y)), "cv_f1": cv_f1}
```

- [ ] **Step 5: Write `scripts/train_fusion.py`**

```python
import argparse
import json

from nba_highlights.training.train import train_model


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the learned fusion model")
    ap.add_argument("--pair", nargs=2, action="append", metavar=("FEATURES", "TRUTH"),
                    required=True, help="features.jsonl truth.json for one game (repeatable)")
    ap.add_argument("--out", required=True, help="output model path (.joblib)")
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()
    metrics = train_model([tuple(p) for p in args.pair], args.out, args.iou)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_train_fusion.py -v`
Expected: 1 passed. Then confirm the default suite is unaffected without the extra:
`.venv/bin/python -m pytest -q` → the train test SKIPS if sklearn absent (it won't be, since installed, but the `importorskip` guarantees dependency-free machines skip it).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/nba_highlights/training/train.py scripts/train_fusion.py tests/test_train_fusion.py
git commit -m "feat: learned-fusion training (LogisticRegression + leave-one-game-out CV)"
```

---

### Task 8: CLI — feature emission, fusion selection, label command

**Files:**
- Modify: `src/nba_highlights/cli.py`
- Test: `tests/test_cli_m2.py`

**Interfaces:**
- Consumes: `nba_highlights.scoring.factory.load_scorer`, `nba_highlights.eval.{GroundTruth, TruthSegment}`, `nba_highlights.models.Report`, `nba_highlights.config.Settings`.
- Produces:
  - `analyze` gains options: `--emit-features PATH` (passed to `run_pipeline(emit_features_path=...)`), `--fusion weighted|learned`, `--fusion-model PATH`. It builds the scorer via `load_scorer(settings.fusion, settings.fusion_model, settings.weights)` and passes it as `scorer=`.
  - New `label` command: `label(report: str, out: str, min_score: float = 0.4)` — reads a `Report`, writes a `GroundTruth` candidate truth JSON containing a `TruthSegment(start_s, end_s)` for every highlight with `score >= min_score`. (A human then prunes the file.)

- [ ] **Step 1: Write the failing test**

```python
import json
from typer.testing import CliRunner
from nba_highlights.cli import app
from nba_highlights.models import (
    Report, VideoInfo, Highlight, SignalBreakdown, AudioSignals, ActionSignals, Evidence,
)

runner = CliRunner()


def _report(spans_scores):
    hs = [Highlight(id=i, start_s=s, end_s=e, score=sc,
                    signals=SignalBreakdown(audio=AudioSignals(), action=ActionSignals()),
                    evidence=Evidence(), member_scenes=[i])
          for i, (s, e, sc) in enumerate(spans_scores)]
    return Report(video=VideoInfo(source="u"), config={}, generated_at="t", highlights=hs)


def test_label_seeds_truth_from_report(tmp_path):
    rp = tmp_path / "r.json"; op = tmp_path / "truth.json"
    rp.write_text(_report([(10.0, 20.0, 0.9), (50.0, 55.0, 0.3)]).model_dump_json())
    result = runner.invoke(app, ["label", "--report", str(rp), "--out", str(op),
                                 "--min-score", "0.4"])
    assert result.exit_code == 0, result.output
    truth = json.loads(op.read_text())
    assert truth["highlights"] == [{"start_s": 10.0, "end_s": 20.0}]  # 0.3 dropped


def test_analyze_passes_emit_and_scorer(tmp_path, monkeypatch):
    import nba_highlights.cli as climod
    captured = {}

    def fake_run_pipeline(source, settings, transcriber, action, **kw):
        captured["emit"] = kw.get("emit_features_path")
        captured["scorer"] = kw.get("scorer")
        return Report(video=VideoInfo(source=source), config={}, generated_at="t",
                      highlights=[])

    monkeypatch.setattr(climod, "run_pipeline", fake_run_pipeline)
    out = tmp_path / "r.json"
    result = runner.invoke(app, ["analyze", "/v.mp4", "--out", str(out),
                                 "--workdir", str(tmp_path),
                                 "--emit-features", str(tmp_path / "f.jsonl")])
    assert result.exit_code == 0, result.output
    assert captured["emit"] == str(tmp_path / "f.jsonl")
    assert captured["scorer"] is not None  # a scorer was built and passed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cli_m2.py -v`
Expected: FAIL (`No such command 'label'` / unknown option `--emit-features`).

- [ ] **Step 3: Edit `src/nba_highlights/cli.py`**

Add imports:
```python
from nba_highlights.eval import GroundTruth, TruthSegment, score_report
from nba_highlights.scoring.factory import load_scorer
```
(Keep the existing `from nba_highlights.eval import GroundTruth, score_report` — merge into the line above so `TruthSegment` is included and there is no duplicate import.)

Add the three options to `analyze`'s signature (after `full_action`):
```python
    emit_features: str = typer.Option(None, help="Write per-shot features JSONL to PATH"),
    fusion: str = typer.Option(None, help="weighted | learned"),
    fusion_model: str = typer.Option(None, help="Path to a trained fusion model (.joblib)"),
```

In `analyze`, after the existing override block and before `settings.check()` (or alongside the other overrides), add:
```python
    if fusion is not None:
        settings.fusion = fusion
    if fusion_model is not None:
        settings.fusion_model = fusion_model
```

Replace the `run_pipeline(...)` call so it builds and passes the scorer + emit path:
```python
    scorer = load_scorer(settings.fusion, settings.fusion_model, settings.weights)
    report = run_pipeline(
        source, settings, transcriber, action,
        workdir=workdir,
        generated_at=datetime.now(timezone.utc).isoformat(),
        title="",
        scorer=scorer,
        emit_features_path=emit_features,
    )
```

Add the `label` command after `eval`:
```python
@app.command()
def label(
    report: str = typer.Option(..., help="Path to a report.json to seed candidates from"),
    out: str = typer.Option(..., help="Output candidate truth JSON"),
    min_score: float = typer.Option(0.4, help="Keep highlights with score >= this"),
):
    with open(report) as f:
        rep = Report.model_validate_json(f.read())
    segs = [TruthSegment(start_s=h.start_s, end_s=h.end_s)
            for h in rep.highlights if h.score >= min_score]
    gt = GroundTruth(highlights=segs)
    with open(out, "w") as f:
        f.write(gt.model_dump_json(indent=2))
    typer.echo(f"wrote {out} ({len(segs)} candidate highlights to prune)")
```

- [ ] **Step 4: Run tests + lint**

Run:
```bash
.venv/bin/python -m pytest tests/test_cli_m2.py tests/test_cli.py -v
.venv/bin/ruff check src tests scripts
```
Expected: new + existing CLI tests pass; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/nba_highlights/cli.py tests/test_cli_m2.py
git commit -m "feat: CLI emit-features, fusion selection, and label command"
```

---

### Task 9: Docs — README + M2 runbook

**Files:**
- Modify: `README.md`
- Create: `docs/M2_LEARNED_FUSION_RUNBOOK.md`

**Interfaces:**
- Consumes: everything.
- Produces: README section documenting the M2 workflow; a runbook with the end-to-end label→train→use loop.

- [ ] **Step 1: Add an M2 section to `README.md`** (after the Evaluate section)

````markdown
## Learned fusion (M2)

The default fusion is hand-weighted. To train a data-driven highlight classifier:

```bash
# 1. Emit per-shot features while analyzing labeled games
highlights analyze GAME1.mp4 --out r1.json --emit-features f1.jsonl
# 2. Seed candidate labels from a run, then hand-prune the JSON to real highlights
highlights label --report r1.json --out truth1.json --min-score 0.4
# 3. Train (needs the train extra: pip install -e '.[train]')
python scripts/train_fusion.py --pair f1.jsonl truth1.json \
                               --pair f2.jsonl truth2.json --out fusion.joblib
# 4. Use the learned model
highlights analyze GAME3.mp4 --out r3.json --fusion learned --fusion-model fusion.joblib
# 5. Measure against held-out truth
highlights eval --report r3.json --truth truth3.json --out eval3.json
```

`--fusion learned` falls back to weighted fusion if the model path is missing.
Feature columns are frozen in `nba_highlights.features.FEATURE_NAMES`.
````

- [ ] **Step 2: Write `docs/M2_LEARNED_FUSION_RUNBOOK.md`**

```markdown
# M2 Runbook — Labeled Evaluation + Learned Fusion

## Install
```
.venv/bin/pip install -e '.[dev,media,audio,asr,action,train]'
```

## Loop
1. **Emit features** for each game you will label:
   `highlights analyze GAME.mp4 --out report.json --emit-features feats.jsonl`
   (add `--action-backend videomae --asr-backend faster-whisper` for real signals)
2. **Seed + prune labels:** `highlights label --report report.json --out truth.json`
   then edit `truth.json` — drop non-highlights, adjust `start_s`/`end_s`.
   Aim for 3–5 fully labeled games.
3. **Train:** `python scripts/train_fusion.py --pair feats1.jsonl truth1.json
   --pair feats2.jsonl truth2.json --out fusion.joblib`
   Prints `{"n", "positives", "cv_f1"}` — `cv_f1` is mean leave-one-game-out F1.
4. **Apply:** `highlights analyze GAME.mp4 --fusion learned --fusion-model fusion.joblib --out r.json`
5. **Measure:** `highlights eval --report r.json --truth held_out_truth.json`

## Notes
- Feature vector order is frozen in `nba_highlights.features.FEATURE_NAMES`;
  retrain if it ever changes.
- `LearnedScorer` falls back to `WeightedScorer` when the model path is absent,
  so the default path never breaks.
- Start with logistic regression (interpretable, tiny-data friendly). Swap in
  `HistGradientBoostingClassifier` in `training/train.py` once you have more
  labeled games.
```

- [ ] **Step 3: Verify docs + full suite**

Run:
```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests scripts
```
Expected: all tests pass (train test skips or passes depending on extra); ruff clean.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/M2_LEARNED_FUSION_RUNBOOK.md
git commit -m "docs: M2 learned-fusion README section + runbook"
```

---

## Self-Review

**1. Spec coverage (against `2026-07-14-m2-learned-fusion-design.md`):**
- §3 labeling workflow (seed from low-threshold run → truth JSON) → Task 8 `label` command. ✔
- §3/§4.2 per-shot feature emission (frozen vector) → Task 1 (`ShotFeatures`/`FEATURE_NAMES`/`to_vector`) + Task 5 (emit). ✔
- §4.1 scorer seam (`HighlightScorer`, `WeightedScorer` default, `LearnedScorer`, `FUSION=weighted|learned`, fallback) → Tasks 2, 3, 4, 5, 8. ✔
- §4.3 feature↔label IoU join + training table → Task 6. ✔
- §4.4 logistic regression + leave-one-game-out CV, persisted artifact → Task 7. ✔
- §6 real P/R/F1 on held-out game → reuses M1 `eval` harness; runbook Task 9 step 5. ✔
- §7 sklearn/joblib in `train` extra, lazy-imported, default suite dependency-free → Task 7 extra + `importorskip`; `LearnedScorer` injectable model avoids joblib in its test (Task 3). ✔
- §4.1 two-pass gate stays on `audio_composite` → Task 5 keeps `comp` as the gate. ✔

**2. Placeholder scan:** No TBD/TODO/"handle edge cases"/"similar to". Every code step has full code. ✔

**3. Type consistency:** `ShotFeatures`/`to_vector`/`FEATURE_NAMES` (Task 1) used identically in scoring (2,3), pipeline (5), dataset (6), train (7). `HighlightScorer.score(feats)->float` signature consistent across WeightedScorer/StubScorer/LearnedScorer (2,3) and the pipeline's `scorer.score(feats)` call (5). `load_scorer(fusion, model_path, weights)` (3) matches the CLI call (8). `Settings.fusion`/`fusion_model` (4) consumed in CLI (8) and recorded in the pipeline config dict (5). `build_table -> (X, y, groups)` (6) consumed by `train_model` (7). `GroundTruth`/`TruthSegment`/`iou` from M1 `eval` used in Task 6 join and Task 8 label. ✔
