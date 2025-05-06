from imblearn.pipeline      import Pipeline as ImbPipeline
from imblearn.combine import SMOTETomek 
from sklearn.compose        import ColumnTransformer
from sklearn.preprocessing  import StandardScaler, FunctionTransformer
from sklearn.calibration    import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.pipeline import Pipeline 
import numpy as np

# ------------------------------------------------------------------ estimators
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model     import LogisticRegression
from sklearn.svm              import SVC
from sklearn.naive_bayes      import GaussianNB
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.neural_network   import MLPClassifier

def _make_estimator(model_type: str, random_state: int):
    if model_type == "RandomForest":
        return RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=random_state,
            class_weight="balanced"                       
        )
    if model_type == "GradientBoosting":
        return GradientBoostingClassifier(random_state=random_state)
    if model_type == "LogisticRegression":
        return LogisticRegression(max_iter=3000, solver="saga",
                                  n_jobs=-1, class_weight="balanced")  
    if model_type == "SVM":
        return SVC(kernel="rbf", probability=False,              
                   class_weight="balanced", random_state=random_state)
    if model_type == "NaiveBayes":
        return GaussianNB()
    if model_type == "KNN":
        return KNeighborsClassifier()
    if model_type == "ANN":
        return MLPClassifier(hidden_layer_sizes=(64, 32),
                             max_iter=300, random_state=random_state)
    raise ValueError(f"Model '{model_type}' not supported.")

# ------------------------------------------------------------------ pipeline
def build_pipeline(model_type: str,
                   random_state: int,
                   smote_ratio: float = 0.10,
                   calibrate: bool   = False):
    # ── column-level preprocessing (log-scale Amount only) ───────────────
    log_amt = FunctionTransformer(lambda x: np.log1p(x))
    amt_pipe = Pipeline([("log", log_amt),
                         ("scale", StandardScaler())])

    cols_amt = ["Amount"]
    cols_rest = [c for c in (f"V{i}" for i in range(1, 29))
                 if c != "Amount" and c != "Time"]

    pre = ColumnTransformer(
        transformers=[("amt", amt_pipe, cols_amt)],
        remainder="passthrough"
    )

    needs_scale = model_type in {"LogisticRegression", "SVM", "KNN", "ANN"}
    needs_smote = model_type not in {"RandomForest", "GradientBoosting"}

    steps = [("pre", pre)]

    # extra global scaling (for non-tree models) --------------------------
    if needs_scale:
        steps.append(("scale_all", StandardScaler(with_mean=False)))  # sparse-safe

    # imbalanced learning --------------------------------------------------
    if needs_smote:
        steps.append(("smote", SMOTETomek(random_state=random_state,
                                          sampling_strategy=smote_ratio)))

    # base estimator -------------------------------------------------------
    base_clf = _make_estimator(model_type, random_state)

    # probability calibration if asked or needed (SVM w/out probas) -------
    if calibrate or not hasattr(base_clf, "predict_proba"):
        base_clf = CalibratedClassifierCV(
            #base_estimator=base_clf,    ## Comment this line out if using SVM 
            method="isotonic", cv=3, ensemble=False
        )

    steps.append(("clf", base_clf))
    return ImbPipeline(steps, verbose=False)