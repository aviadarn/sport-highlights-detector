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
- Keep `AUDIO_GATE` (the two-pass action gate) the SAME between the
  `--emit-features` run you train on and the `--fusion learned` run you serve
  with. `action_score` is 0.0 for shots below the gate, so a different gate
  shifts the feature distribution and the learned model sees out-of-distribution
  inputs. Retrain if you change the gate.
