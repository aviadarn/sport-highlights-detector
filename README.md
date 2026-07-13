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
