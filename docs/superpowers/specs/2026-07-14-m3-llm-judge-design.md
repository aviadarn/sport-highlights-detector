# M3 — Multimodal LLM Highlight Judge (Design)

**Date:** 2026-07-14
**Status:** Proposed (next milestone). Builds on M1 (detector) and M2
(labeled evaluation + learned fusion).

## 1. Motivation

M2 shipped the learned-fusion loop but exposed the real bottleneck: **labels.**
The only labels we have are an AI transcript proxy (one broadcast, 5 windows),
and the CV result was fragile precisely because the labeled set was tiny. Human
labeling is the correct ground truth but doesn't scale — nobody is going to
hand-mark 50 games.

An **LLM judge** automates and scales what the proxy did by hand. Critically, a
**multimodal (vision) judge** — one that looks at the scene's keyframes, not
just the transcript — labels from a modality that is *independent* of the
audio+action features the fusion model trains on. That independence is the whole
point: training audio/action features to predict visually-grounded labels is a
real learning task, not the near-tautology a transcript-only judge would create
(transcript keywords already overlap the `keyword_count` feature).

**This is a data + evaluation tool, not a detector change.** The judge produces
`GroundTruth` JSON that drops into the existing M2 `label → train → eval` loop.
The runtime detector (M1 pipeline + M2 scorer) is untouched.

## 2. Goal

Given a detector run's candidate segments, have a Claude vision model judge each
one (highlight or not, + play type + confidence) from its keyframes and
commentary, emitting the standard `GroundTruth` truth JSON — at a scale and
consistency no human matches — and calibrate that judge against a small
human-labeled anchor set so its labels are trustworthy enough to train on and to
approximate evaluation with.

## 3. Two distinct uses — keep them separate

1. **Training labels (primary, noise-tolerant).** Judge many games cheaply →
   large weakly-labeled training set for the M2 learned fusion. Scale beats
   cleanliness for a small classifier. This is where the judge pays off.
2. **Evaluation ground truth (secondary, must be anchored).** The judge can
   *approximate* precision/recall on held-out games, but a model grading a model
   is only trustworthy once validated. **A small human-labeled holdout remains
   the authoritative eval number.** The judge scales measurement; it does not
   replace the human anchor.

## 4. Architecture

### 4.1 `HighlightJudge` seam (same pattern as M1 backends)

Note: nba-highlights has **no Claude integration yet** — this is the first. It
follows the established injectable-backend + stub pattern.

```python
class HighlightJudge(Protocol):
    def judge(self, clip: JudgeInput) -> JudgeVerdict: ...
```

- `JudgeInput{scene_id, start_s, end_s, keyframe_paths: list[str], transcript: str}`.
- `JudgeVerdict{is_highlight: bool, confidence: float, play_type: str, rationale: str}`
  (pydantic).
- Backends: `stub` (deterministic, default — keeps the test suite offline) and
  `claude` (`ClaudeHighlightJudge`).

### 4.2 The Claude backend (multimodal + structured output)

- **Input:** for each candidate segment, sample a few keyframes (reuse
  `media.frame_times` + `media.extract_frame`, already in M1) and send them as
  **base64 image content blocks** ahead of a text block carrying the transcript
  and the ask. Vision is native on the Claude 4 family.
- **Output:** use structured outputs — `client.messages.parse(model=...,
  output_format=JudgeVerdict)` returns a validated `JudgeVerdict`, so there's no
  brittle JSON parsing.
- **Model:** default `claude-opus-4-8` (strong multimodal reasoning). Selectable
  via env for cost/scale: `claude-sonnet-4-6` (~½ cost) or `claude-haiku-4-5`
  (cheapest) for bulk labeling once the prompt is validated on Opus. Adaptive
  thinking (`thinking={"type":"adaptive"}`) with modest `effort` for the
  judgment; `budget_tokens` is not used on 4.x.
- **Prompt:** a fixed system prompt defining "highlight" for NBA (made
  baskets/dunks/threes, blocks, steals, big defensive plays, buzzer-beaters —
  NOT timeouts, replays-of-nothing, analysis, or dead-ball talk), returning the
  verdict per candidate. Frozen and versioned so labels are reproducible.
- **SDK:** official `anthropic` SDK via a new lazy-imported `judge` extra;
  `ANTHROPIC_API_KEY` from the environment.

### 4.3 Data flow

```
detector run (low threshold) ─▶ candidates + keyframes + transcript
        │
        ▼
  HighlightJudge.judge(each candidate)         ← Claude vision, structured verdict
        │
        ├─▶ keep is_highlight=true (optionally confidence ≥ τ) as TruthSegments
        ▼
   GroundTruth JSON  ──▶  M2 `train_fusion` / `eval` (unchanged)
```

A new `highlights judge --report R.json --workdir W --out truth.json
[--backend claude|stub] [--min-confidence 0.6]` command reads a report, judges
each highlight's segment, and writes the `GroundTruth` file the M2 loop already
consumes. It replaces the manual `highlights label` prune step with an automated
one; the human still edits the output when anchoring (§5).

### 4.4 Scale + cost controls

- **Only candidates are judged** (the detector's gated output), not every shot —
  bounded call count (~tens–low-hundreds per game).
- **Batch API** (`messages.batches`, 50% cost) for offline bulk labeling of many
  games; interactive `judge` for a single game.
- Rough cost with Opus 4.8 and ~3 keyframes/candidate: order of a few dollars per
  full game; Sonnet/Haiku and Batch cut that severalfold. Recorded per run.

## 5. Judge validation — the non-negotiable step

Before trusting judge labels at scale:

1. Human-label a small anchor set (1–2 games, the M2 `label` command + manual
   prune — real human judgment).
2. Run the judge on the same games.
3. Report **agreement** (precision/recall/F1 of judge vs human, and Cohen's κ) —
   reuse `eval.iou` for segment matching. A new `highlights judge-agreement
   --judged J.json --human H.json` command.
4. Only if agreement clears a bar (e.g. F1 ≥ 0.8) do we use the judge to label
   the rest. Otherwise iterate the prompt/model/keyframe sampling.

Keep one human-labeled game entirely out of training as the authoritative eval
holdout — the judge never grades the final number.

## 6. Success criteria

- Judge-vs-human agreement measured and reported (we have no such number today).
- With judge-labeled games feeding M2, the learned fusion's held-out F1
  (against the **human** holdout) beats both the weighted baseline and the
  proxy-labeled M2 result.
- The whole judging path stays behind the `judge` extra; the default `[dev]`
  suite runs offline on the stub.

## 7. Testing

- **Unit (default suite, offline):** `StubHighlightJudge` returns deterministic
  verdicts; test the report→GroundTruth conversion and the confidence filter with
  it. No `anthropic` import in the default suite.
- **`ClaudeHighlightJudge`:** unit-test the request assembly (image blocks +
  transcript + system prompt) and verdict mapping against a **fake client**
  (injected), so no network/model. A `slow`-marked live smoke test on one clip
  (needs `ANTHROPIC_API_KEY`) is optional and deselected by default.
- **Agreement command:** tested on synthetic judged/human fixtures via `eval.iou`.
- `anthropic` lives in the `judge` extra, lazy-imported inside the Claude backend
  (same discipline as torch/faster-whisper).

## 8. Out of scope (future)

- Fine-tuning VideoMAE on basketball actions (the other M2-deferred lever;
  independent of this).
- Scoreboard OCR as a feature/label signal.
- Using the judge as a *runtime* scorer (it's a labeling/eval tool; the runtime
  path stays the M2 learned fusion).
- Active-learning loop (judge proposes, human reviews only disagreements) — a
  natural follow-on once agreement is measured, but not in this milestone.
