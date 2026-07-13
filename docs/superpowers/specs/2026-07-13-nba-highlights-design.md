# NBA Highlights Detector — Design

**Date:** 2026-07-13
**Status:** Approved (brainstorming); pending implementation plan.

## 1. Goal

Given a **YouTube URL of a full NBA game**, produce a **JSON report** listing
**play-level highlight segments** with timestamps, a fused highlight score, and
per-signal evidence. Highlights are detected by combining **audio cues** (crowd
loudness, commentator excitement, hype-keyword spotting) with **action
recognition** (pretrained video model). No video is rendered — the deliverable
is the JSON report only.

This is a **fresh standalone project** (`sport-highlights-detector/`) with an
**in-process CLI** pipeline. It reuses proven components and patterns from the
sibling `celebvision` project (scene detection, faster-whisper ASR, the
pluggable-backend protocol pattern, the precision/recall eval harness, CLI
shape) but does not depend on its async infrastructure (Kafka/Postgres/MinIO).

### Decisions locked in brainstorming

| Question | Decision |
|---|---|
| Deliverable | JSON report only (timestamps + scores + evidence). No rendered clips/reel. |
| Input (first target) | YouTube URL, full game (~2.5 hr), fetched via yt-dlp. |
| Action recognition | Pretrained video model (VideoMAE, Kinetics-400) on sampled scene clips. Stub backend kept for tests. |
| Audio cues | All three: crowd roar/loudness, commentator excitement (prosody), keyword spotting (ASR + hype lexicon). |
| Selection | Weighted fusion score (0–1) + threshold. Every scene scored; output those ≥ threshold. |
| Footprint | Fresh standalone repo, in-process CLI. Reuse celebvision pieces by copy/adapt. |
| Detection atom | Scene-detect + merge: score PySceneDetect shots, merge adjacent high-scoring shots into play-level segments. One JSON entry = one play. |

## 2. Architecture

Linear in-process pipeline, one video at a time:

```
YouTube URL
  ─▶ ingest        (yt-dlp → MP4, ffmpeg → 16 kHz mono WAV)
  ─▶ scene detect  (PySceneDetect ContentDetector → shots[start_s, end_s])
  ─▶ audio scoring (per shot: loudness + prosody + keywords)   ┐
  ─▶ action scoring(per candidate shot: VideoMAE Kinetics)     ┘
  ─▶ fuse          (weighted audio + action → shot score 0–1)
  ─▶ merge         (adjacent high shots, gap ≤ max_gap → play segments)
  ─▶ select        (segment score ≥ threshold)
  ─▶ Report JSON
```

### Full-game cost strategy — two-pass

Processing a ~2.5 hr broadcast with a video action model is the dominant cost,
especially on CPU. The pipeline is **two-pass**:

1. **Pass 1 (cheap):** score every shot on audio signals only (loudness,
   prosody, keywords). Audio scoring is CPU-cheap.
2. **Pass 2 (expensive):** run the video action model **only on shots that pass
   a low audio pre-gate** (candidate shots). Most of a game is quiet; the gate
   removes the bulk of shots, cutting video-model calls roughly 10×.

`--full-action` overrides this to score every shot with the action model.

**Known trade-off:** a visually strong but audio-silent play will not receive an
action score under the default gate and is unlikely to be selected. This is
accepted for NBA, where notable plays reliably draw a crowd/commentator
response. The gate threshold is configurable, and `--full-action` disables the
gate entirely.

## 3. Components and interfaces

Pluggable backends sit behind protocols (the celebvision pattern), each with a
deterministic **stub** default so the full pipeline runs in unit tests without
downloading models.

- `ActionRecognizer.score_clip(video_path, start_s, end_s) -> ActionResult`
  - `ActionResult{ score: float (0–1), label: str, top_k: list[(label, prob)] }`
  - Backends: `stub` (deterministic), `videomae` (HuggingFace VideoMAE fine-tuned
    on Kinetics-400). `score` = max probability over basketball-relevant Kinetics
    classes (`dunking basketball`, `shooting basketball`, `playing basketball`,
    `dribbling basketball`); `label` = top such class.
- `Transcriber.transcribe(wav_path) -> Transcript`
  - `Transcript{ segments: [{start_s, end_s, text, words:[{text,start_s,end_s}]}] }`
  - Backends: `stub`, `faster-whisper` (adapted from celebvision; model size
    configurable, default `small`, CPU int8).
- **Audio scorers** are pure functions over the extracted WAV (no protocol
  needed; fully testable with synthetic audio):
  - `loudness`: short-time RMS energy per shot (peak + sustained), percentile-
    normalized across the game → 0–1.
  - `prosody`: commentator excitement from librosa fundamental-frequency (pyin)
    and energy rise per shot → 0–1.
  - `keywords`: intersect ASR word timestamps within a shot window against a
    configurable hype lexicon (e.g. `dunk`, `three`, `and one`, `buzzer`,
    `oh my`), count-weighted → 0–1.

Backend selection is env/CLI-driven (celebvision style): `ACTION_BACKEND`,
`ASR_BACKEND`.

## 4. Data flow and scoring

Per shot:

```
audio_composite = wL·loudness + wP·prosody + wK·keywords
fused           = wA·audio_composite + wV·action
```

All weights (`wL, wP, wK, wA, wV`), the selection `threshold`, the audio
pre-gate, and `max_gap_s` live in `config.py` (pydantic-settings,
env-overridable). Signal scores are **percentile-normalized per game** so
thresholds are game-relative rather than absolute (a loud arena and a quiet one
are comparable).

**Merge:** consecutive shots with `fused ≥ threshold` whose inter-shot gap is
`≤ max_gap_s` (default 3 s) collapse into one `HighlightSegment`. Segment score
= max of member shot scores. Evidence is aggregated across members (peak
loudness time, union of matched keywords, top action label, member scene ids).

**Select:** segments with `score ≥ threshold` are emitted, ordered by start
time.

## 5. Output JSON schema

```json
{
  "video": {"source": "https://youtube.com/watch?v=...", "title": "...", "duration_s": 8734.2},
  "config": {"weights": {"wL": 0.4, "wP": 0.3, "wK": 0.3, "wA": 0.5, "wV": 0.5},
             "threshold": 0.5, "action_backend": "videomae", "asr_model": "small"},
  "generated_at": "2026-07-13T00:00:00Z",
  "highlights": [
    {
      "id": 0,
      "start_s": 1423.5,
      "end_s": 1431.2,
      "score": 0.87,
      "signals": {
        "audio": {"loudness": 0.9, "prosody": 0.8, "keywords": 0.6},
        "action": {"score": 0.82, "label": "dunking basketball"}
      },
      "evidence": {
        "peak_loudness_s": 1425.1,
        "keywords": ["dunk", "and one"],
        "transcript": "..."
      },
      "member_scenes": [142, 143, 144]
    }
  ]
}
```

Modeled with pydantic in `models.py` (`Shot`, `ShotScore`, `HighlightSegment`,
`Report`).

## 6. Error handling

- yt-dlp / ffmpeg failures raise a clear `IngestError`; missing `ffmpeg` on the
  host is detected upfront with an actionable message.
- Per-shot scoring is wrapped: one shot that fails action or audio scoring is
  logged and scored 0, never aborting the run.
- ASR failure degrades gracefully — `keywords` contributes 0 and the pipeline
  proceeds on loudness + prosody + action.
- Configuration errors (bad weights, unknown backend) fail loud at startup.

## 7. Testing and evaluation

- **Unit tests** run the full pipeline on stub backends (deterministic, no model
  downloads, no network).
- **Pure-function tests** cover `fuse`, `merge`, `select`, and each audio scorer
  (synthetic audio / synthetic shot scores).
- **Integration test** (`slow` marker, deselected by default) runs real models
  on a short local clip.
- **Eval harness** — `highlights eval --report R --truth T`: matches detected
  segments to a hand-labeled truth JSON by temporal IoU ≥ 0.5 and reports
  precision / recall / F1. This is the correctness measure, since full-game
  auto-eval requires labels. Adapted from celebvision's eval scorer.

## 8. Repository layout and dependencies

```
src/nba_highlights/
  models.py interfaces.py config.py errors.py
  media.py            # yt-dlp ingest, ffmpeg audio extract, frame sampling
  scenes.py           # PySceneDetect wrapper
  audio/{loudness,prosody,keywords}.py
  action/{base,stub,videomae_client}.py
  transcribe/{base,stub,faster_whisper_client}.py
  fuse.py merge.py select.py
  pipeline.py         # orchestrate end-to-end
  eval.py cli.py
tests/
docs/superpowers/specs/2026-07-13-nba-highlights-design.md
pyproject.toml
```

Dependencies split by optional extra (install only what a run needs):

| Extra | Packages | Used for |
|---|---|---|
| `media` | yt-dlp, scenedetect, ffmpeg-python | ingest + scene detection |
| `audio` | librosa, numpy | loudness + prosody scoring |
| `action` | torch, transformers | VideoMAE action recognition |
| `asr` | faster-whisper | keyword spotting transcription |
| `dev` | pytest, ruff | tests + lint |

Python 3.11+ (developed on 3.13). Stub backends require none of the heavy
extras, so `pip install -e '.[dev]'` is enough to run the test suite.

## 9. CLI

```bash
highlights analyze "https://www.youtube.com/watch?v=<id>" \
  --out report.json --workdir ./work \
  [--threshold 0.5] [--action-backend videomae|stub] \
  [--asr-backend faster-whisper|stub] [--full-action]

highlights eval --report report.json --truth truth.json --out eval.json
```

## 10. Out of scope (first milestone)

- Rendered highlight clips or stitched reels (JSON only).
- Sports other than NBA (architecture is sport-agnostic, but lexicon + action
  classes are NBA-tuned first).
- Async / multi-job service infrastructure.
- GPU-specific serving (VideoMAE runs on CPU or GPU via torch device selection;
  no Triton).
