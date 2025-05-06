import os, warnings, numpy as np, matplotlib.pyplot as plt, seaborn as sns
import shap
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve, roc_curve,
                             confusion_matrix, classification_report,
                             fbeta_score)
import matplotlib
from sklearn.calibration import calibration_curve 

warnings.filterwarnings("ignore", category=UserWarning)
os.makedirs("results", exist_ok=True)
matplotlib.use("Agg") 


# ---------------------------------------------------------------- evaluation
def evaluate_model(model, X_test, y_test, model_name="model", shap_sample_size=1000):
    # --- predictions / scores -------------------------------------------
    y_pred   = model.predict(X_test)
    try:
        y_scores = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        y_scores = model.decision_function(X_test)
        y_scores = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())

    # --- scalar metrics --------------------------------------------------
    roc_auc = roc_auc_score(y_test, y_scores)
    auprc   = average_precision_score(y_test, y_scores)
    f2      = fbeta_score(y_test, y_pred, beta=2)

    rpt = classification_report(y_test, y_pred, output_dict=True)
    prec_pos = rpt["1"]["precision"]
    rec_pos  = rpt["1"]["recall"]

    # --- plots -----------------------------------------------------------
    fig_paths = {}

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion – {model_name}")
    plt.xlabel("Pred"); plt.ylabel("Actual")
    p = f"results/cm_{model_name}.png"; plt.tight_layout(); plt.savefig(p, dpi=300); plt.close()
    fig_paths["conf_mat"] = p

    # roc
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    plt.figure(figsize=(4, 3))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--"); plt.legend()
    plt.title(f"ROC – {model_name}")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    p = f"results/roc_{model_name}.png"; plt.tight_layout(); plt.savefig(p, dpi=300); plt.close()
    fig_paths["roc"] = p

    # pr
    pr, rc, _ = precision_recall_curve(y_test, y_scores)
    plt.figure(figsize=(4, 3))
    plt.plot(rc, pr)
    plt.title(f"PR – {model_name} (AUPRC={auprc:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    p = f"results/pr_{model_name}.png"; plt.tight_layout(); plt.savefig(p, dpi=300); plt.close()
    fig_paths["pr"] = p

    # calibration
    prob_true, prob_pred = calibration_curve(y_test, y_scores, n_bins=10)
    plt.figure(figsize=(4, 3))
    plt.plot(prob_pred, prob_true, "o-"); plt.plot([0, 1], [0, 1], "k--")
    plt.title(f"Calibration – {model_name}")
    plt.xlabel("Mean Pred Prob"); plt.ylabel("Frac Pos")
    p = f"results/cal_{model_name}.png"; plt.tight_layout(); plt.savefig(p, dpi=300); plt.close()
    fig_paths["cal"] = p

    # SHAP summary (with subsample)
    try:
        shap_sample_size = min(1000, X_test.shape[0])
        X_sampled = X_test.sample(shap_sample_size, random_state=42)

        # Apply preprocessing step only
        preprocessor = model.named_steps["pre"]
        X_transformed = preprocessor.transform(X_sampled)

        final_model = model.named_steps["clf"]
        explainer = shap.Explainer(final_model, X_transformed)
        shap_values = explainer(X_transformed)

        plt.figure()
        shap.summary_plot(shap_values, X_transformed, show=False)
        p = f"results/shap_{model_name}.png"
        plt.tight_layout(); plt.savefig(p, dpi=300); plt.close()
        fig_paths["shap"] = p

    except Exception as e:
        print(f"[!] SHAP skipped for {model_name}: {e}")


    # --- metric dict -----------------------------------------------------
    return {
        "ROC_AUC"   : roc_auc,
        "AUPRC"     : auprc,
        "F2"        : f2,
        "Precision_Pos" : prec_pos,
        "Recall_Pos"    : rec_pos,
        **{k + "_path": v for k, v in fig_paths.items()}
    }