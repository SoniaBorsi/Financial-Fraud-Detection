import os, json, yaml, warnings
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, fbeta_score

from utils.data_loader    import load_data
from utils.preprocess     import preprocess_data
from utils.train_model    import build_pipeline
from utils.evaluate_model import evaluate_model
from sklearn.exceptions import ConvergenceWarning
#warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing")

# ─────────────────────────────────────────────────────────────────────────
def main() -> None:

    # ── read config ──────────────────────────────────────────────────────
    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    model_name   = cfg["model"]
    random_state = cfg.get("random_state", 42)
    test_size    = cfg.get("test_size", 0.30)
    drop_time    = cfg.get("drop_time", True)
    smote_ratio  = cfg.get("smote_ratio", 0.10)
    calibrate    = cfg.get("calibrate", False)
    cv_folds     = cfg.get("cv_folds", 5)

    # ── data split ───────────────────────────────────────────────────────
    print("Loading dataset …")
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(
        df, target="Class",
        test_size=test_size,
        random_state=random_state,
        drop_time=drop_time
    )

    # ── build pipeline ───────────────────────────────────────────────────
    pipe = build_pipeline(
        model_type=model_name,
        random_state=random_state,
        smote_ratio=smote_ratio,
        calibrate=calibrate
    )

    # ── skip CV for these models ───────────────────────────────────
    #skip_cv_models = {"RandomForest", "NaiveBayes"}
    run_cv = model_name #not in skip_cv_models
    cv_metrics = {}

    if run_cv:
        print(f"Running {cv_folds}-fold CV …")
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        scoring = {
            "roc_auc": "roc_auc",
            "auprc"  : "average_precision",
            "f2"     : make_scorer(fbeta_score, beta=2)
        }

        import platform
        is_mac = platform.system() == "Darwin"

        cv_res = cross_validate(
            pipe,
            X_train, y_train,
            cv=skf,
            scoring=scoring,
            return_train_score=False,
            n_jobs=1 if is_mac else -1  
        )

        cv_metrics = {
            f"CV_{k[5:]}": float(np.mean(v))
            for k, v in cv_res.items()
            if k.startswith("test_")
        }
    else:
        print(f"Skipping CV for model → {model_name}")

    # ── final training ───────────────────────────────────────────────────
    print(f"Training final model → {model_name}")
    pipe.fit(X_train, y_train)

    # ── hold-out evaluation ─────────────────────────────────────────────
    test_metrics = evaluate_model(pipe, X_test, y_test, model_name=model_name,
                              shap_sample_size=cfg.get("shap_sample_size", 1000))
    metrics = {**cv_metrics, **test_metrics}

    os.makedirs("results", exist_ok=True)
    out_json = f"results/metrics_{model_name}.json"
    with open(out_json, "w") as fp:
        json.dump(metrics, fp, indent=2)

    # ── report ──────────────────────────────────────────────────────────
    if cv_metrics:
        print("\n── Cross-validation Summary ──")
        for k, v in cv_metrics.items():
            print(f"{k}: {v:.4f}")

    print("\n── Hold-out Evaluation Summary ──")
    for k, v in test_metrics.items():
        if k.endswith("_path"):
            print(f"{k}: {v}")
        else:
            print(f"{k}: {v:.4f}")

    print(f"\nResults saved → {out_json}")

# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()