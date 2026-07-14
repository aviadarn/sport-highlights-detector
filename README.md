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

## LLM judge for labels (M3)

Hand-labeling games doesn't scale. A Claude vision judge labels a detector run's
candidates from their keyframes + commentary, producing the same truth JSON the
learned-fusion loop consumes:

```bash
# 1. Analyze a local game with --emit-features (see Learned fusion), keeping the video
highlights analyze GAME.mp4 --out report.json --emit-features feats.jsonl \
  --action-backend videomae --asr-backend faster-whisper
# 2. Judge the candidates -> GroundTruth (needs the judge extra + ANTHROPIC_API_KEY)
highlights judge --report report.json --video GAME.mp4 --out truth.json \
  --backend claude --model claude-opus-4-8 --min-confidence 0.6
# 3. Calibrate: compare judge labels to a human-labeled game before trusting at scale
highlights judge-agreement --judged truth.json --human human_truth.json --out agree.json
```

`--backend stub` runs offline (no API). The judge writes the same `GroundTruth`
shape as `highlights label`, so `train_fusion.py` and `highlights eval` consume
it unchanged. **Validate agreement against a human-labeled game (F1/κ) before
using judge labels to train, and keep one human-labeled game as the authoritative
eval holdout.**

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
