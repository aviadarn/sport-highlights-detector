# M2 — Labeled Evaluation + Learned Fusion (Design)

**Date:** 2026-07-14
**Status:** Proposed (next milestone). Builds on the M1 detector
(`2026-07-13-nba-highlights-design.md`).

## 1. Motivation

The M1 detector works end-to-end but has two coupled gaps exposed by the first
full-game run (Knicks–Pacers, 118 highlights over 2h21m):

1. **No ground truth.** We cannot state the detector's precision/recall. 118
   highlights could be 40% false positives — unknown and unmeasurable today.
2. **The decision boundary is hand-guessed.** Fusion weights
   (`wL/wP/wK/wA/wV`) and the `0.5` threshold are fixed constants. That is *why*
   the output is uncalibrated (≈1 highlight / 72 s). Nothing about the boundary
   is fit to what a human actually calls a highlight.

These are the same gap: without labels we can neither *measure* nor *learn* the
boundary. M2 closes it — establish measurement, then replace the hand-tuned
boundary with a data-driven one.

**Explicitly the cheapest high-ROI lever first.** Fine-tuning the VideoMAE
action model is a bigger, costlier lever deferred to M3 (see §8); it is not
justified until we can measure whether it helps.

## 2. Goal

Given a handful of human-labeled games, (a) report real precision/recall/F1 for
the detector, and (b) replace the hand-weighted fusion with a small **learned
highlight classifier** that plugs in behind the existing fusion seam, defaulting
to the current weighted logic when no trained model is present.

## 3. Scope

**In:**
- A labeling workflow producing the truth JSON the M1 `eval` harness already
  consumes (`{highlights: [{start_s, end_s}]}`), seeded from a low-threshold run
  so a human prunes candidates rather than scrubbing 2.5 hours blind.
- Per-shot **feature emission** from the pipeline (a frozen feature vector),
  joined to labels by temporal IoU to build a training table.
- A **learned fusion scorer** (`LearnedScorer`) trained offline, persisted, and
  selectable at runtime (`FUSION=weighted|learned`), with `weighted` as the
  default/fallback.
- A **leave-one-game-out evaluation loop** reporting P/R/F1 and a PR curve to
  pick the operating point.

**Out (future — see §8):** VideoMAE fine-tuning on basketball-specific actions
(M3); scoreboard OCR features; crowd-vs-commentary source separation; bigger ASR
model (that is a config change, not a milestone).

## 4. Architecture

### 4.1 Fusion behind a scorer seam

M1 computes `audio_composite` + `fuse` inline in `pipeline.py`. M2 extracts that
into a small seam so the boundary is swappable, exactly like the existing
transcriber/action backends:

```python
class HighlightScorer(Protocol):
    def score(self, feats: ShotFeatures) -> float: ...   # -> 0..1
```

- `WeightedScorer` — wraps the current `fuse.audio_composite` + `fuse.fuse`
  logic. Behavior-identical to M1. Default.
- `LearnedScorer(model_path)` — lazy-loads a persisted scikit-learn model and
  returns its positive-class probability. Selected via `FUSION=learned` +
  `FUSION_MODEL=path`. Falls back to `WeightedScorer` if the model is absent.

`pipeline.py` calls `scorer.score(feats)` per shot instead of the inline
weighted sum. Merge/select/threshold downstream are unchanged (the threshold now
applies to the scorer's output; for the learned model it's a probability, so the
threshold becomes a calibrated operating point rather than a guess).

### 4.2 Frozen feature vector

`ShotFeatures` (pydantic) — kept small and generalizable across broadcasts:

| feature | source | rationale |
|---|---|---|
| `loudness_n` | M1 normalized loudness | crowd roar |
| `prosody_n` | M1 normalized prosody | commentator excitement |
| `keywords_n` | M1 normalized keyword score | hype vocabulary density |
| `keyword_count` | raw distinct matches | absolute, not just percentile |
| `action_score` | VideoMAE basketball prob | on-court action intensity |
| `shot_duration_s` | shot span | replays/isolation cuts differ from live |
| `rel_position` | `start_s / duration_s` | late-game plays are likelier highlights |

Per-game percentile normalization (already in M1) is what lets features transfer
across arenas with different absolute loudness. The generalizable signals
(keywords, action, relative position) carry most of the cross-game weight.

### 4.3 Data flow

```
labeled games ─┐
               ├─▶ join features↔labels by temporal IoU ─▶ training table
pipeline ──emit┘        (scripts/train_fusion.py)              │
  ShotFeatures                                                 ▼
  per shot                                    fit classifier + leave-one-game-out CV
                                                              │
                                              persist model (joblib) + metrics
                                                              │
                                              LearnedScorer loads it at runtime
```

- **Feature emission:** `highlights analyze --emit-features feats.jsonl` writes
  one `ShotFeatures` row per shot (with `start_s`/`end_s`) alongside the report.
- **Label join:** a shot is positive iff it overlaps a truth segment with
  IoU ≥ 0.5 (reusing `eval.iou`).
- **Training:** `scripts/train_fusion.py` reads feature tables + truth JSONs,
  builds the table, fits the model, prints leave-one-game-out CV metrics, writes
  the artifact.

### 4.4 Model choice

Start with **logistic regression** (L2-regularized): tiny data, interpretable
coefficients (you can read off which signal matters), well-calibrated
probabilities for thresholding. **Gradient-boosted trees** (e.g.
`HistGradientBoostingClassifier`) as a drop-in upgrade once there are enough
labeled games to justify it. Both live behind the same `LearnedScorer`; the
artifact records which was used.

## 5. Labeling workflow

`highlights label --report R.json --out truth.json` seeds a candidate file from
a **low-threshold** run (e.g. 0.4), so the human prunes an over-complete list
(keep/drop per candidate + adjust bounds) instead of labeling from scratch. The
output is the standard `GroundTruth` truth JSON — no new format. Target: 3–5
fully labeled games to start.

## 6. Success criteria

- A **real P/R/F1 number** for the detector on a held-out game (we have none
  today).
- Learned fusion beats the best hand-weighted threshold on held-out F1 at a
  comparable, usable highlight count (~30–40/game), verified by leave-one-game-out
  CV.
- `WeightedScorer` remains the zero-dependency default; the learned path is
  purely additive.

## 7. Testing

- **Unit (default suite, no heavy deps):** `WeightedScorer` parity with M1
  fusion; feature↔label IoU join; `LearnedScorer` falls back to weighted when
  the model is absent; a `StubScorer` for deterministic pipeline tests.
- **Training script:** runs on a tiny synthetic fixture (a few feature rows + a
  truth file) and produces a model artifact + metrics — asserted on a
  fixture, not a real game.
- scikit-learn + joblib go in a new optional `train` extra, lazy-imported. The
  default `[dev]` suite never imports them (same discipline as M1's heavy
  backends).

## 8. Out of scope → M3 candidates

Once M2 gives measurement, these become justifiable and rankable by measured
gain:
- **Fine-tune VideoMAE** on basketball-specific actions (dunk/three/block/steal/
  foul) — the biggest quality lever, but needs a labeled action dataset.
- **Scoreboard OCR** — a score change is near-proof of a scoring play; a strong,
  cheap feature.
- **Crowd-vs-commentary source separation** — cleaner loudness.
- **Bigger ASR model** — a config change, folded in whenever.
