import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score,
                             PrecisionRecallDisplay)

results = {}   # one row per evaluated model


def evaluate_model(name, model, X_test, y_test, threshold=0.5, show=True):
    """Evaluate a fitted model (a Pipeline or any model with predict_proba)."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    results[name] = {
        "PR-AUC": average_precision_score(y_test, y_proba),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Threshold": threshold,
    }

    if show:
        print(f"=== {name} (threshold = {threshold}) ===")
        print(classification_report(y_test, y_pred, digits=4))
        print(confusion_matrix(y_test, y_pred))
        print("ROC-AUC:", round(results[name]["ROC-AUC"], 4))
        print("PR-AUC :", round(results[name]["PR-AUC"], 4))
    return y_proba


def compare_results():
    """Table of all evaluated models, best PR-AUC first."""
    return (pd.DataFrame(results).T
              .sort_values("PR-AUC", ascending=False)
              .round(4))


def plot_pr_curves(models, X_test, y_test, save_path=None):
    """models: dict {name: fitted_model}. Draws all PR curves on one plot."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, model in models.items():
        PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    ax.axhline(y_test.mean(), ls="--", color="gray", label="Useless model")
    ax.set_title("Precision-Recall curves")
    ax.legend()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.show()
    