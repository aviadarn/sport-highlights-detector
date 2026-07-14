# M3 Runbook — Multimodal LLM Highlight Judge

## Install
```
.venv/bin/pip install -e '.[dev,media,audio,asr,action,train,judge]'
export ANTHROPIC_API_KEY=...
```

## Loop
1. **Analyze** each game locally, keeping the video file, with `--emit-features`.
2. **Judge:** `highlights judge --report report.json --video GAME.mp4 --out truth.json --backend claude`
   - Model: `--model claude-opus-4-8` (default), or `claude-sonnet-4-6` / `claude-haiku-4-5` for cheaper bulk labeling once the prompt is validated on Opus.
   - `--min-confidence` filters low-confidence verdicts.
3. **Calibrate (required before trusting):** human-label 1-2 games (`highlights label` + manual prune), then
   `highlights judge-agreement --judged truth.json --human human_truth.json`.
   Only scale judging if F1 clears your bar (e.g. ≥ 0.8).
4. **Train + measure:** feed judge-labeled games into `scripts/train_fusion.py`; keep one
   human-labeled game entirely out of training as the authoritative `highlights eval` holdout.

## Notes
- The judge labels from the VIDEO (keyframes), a modality independent of the
  audio/keyword features the fusion model trains on — that independence is why
  the labels are useful and not circular.
- `anthropic` lives in the `judge` extra, lazy-imported; the default `[dev]`
  suite runs offline on `--backend stub`.
- Cost is bounded (only gated candidates are judged). For many games, use the
  Batch API path (50% cost) — a future addition; the current command judges
  interactively.
- A judge is a proxy for human judgment, not human judgment: it scales labeling
  and approximates eval, but the authoritative eval number stays human-anchored.
