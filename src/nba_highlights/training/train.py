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
            clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
            clf.fit([X[i] for i in train_idx], y_train)
            preds = clf.predict([X[i] for i in test_idx])
            scores.append(f1_score([y[i] for i in test_idx], preds,
                                   zero_division=0))
        cv_f1 = sum(scores) / len(scores) if scores else None

    model = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    model.fit(X, y)
    joblib.dump(model, out_path)
    return {"n": len(y), "positives": int(sum(y)), "cv_f1": cv_f1}
