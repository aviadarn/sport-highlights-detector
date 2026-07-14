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
