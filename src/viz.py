import os
import numpy as np
import matplotlib.pyplot as plt
from .config import FIG_DIR

def plot_roc_curves(roc_list, title="ROC Curves"):
    fig = plt.figure()
    for roc in roc_list:
        plt.plot(roc.fpr, roc.tpr, label=f"{roc.label} (AUC={roc.auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, "roc_curves.png"), dpi=200, bbox_inches="tight")
    return fig

def plot_feature_importance(model, feature_names, title="Feature Importance (Top 15)", top_k=15):
    if hasattr(model, "coef_"):
        importance = np.abs(model.coef_).ravel()
    elif hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    else:
        raise ValueError("Model has no coef_ or feature_importances_")

    idx = np.argsort(importance)[::-1][:top_k]
    names = [feature_names[i] for i in idx]
    vals = importance[idx]

    fig = plt.figure()
    plt.barh(names[::-1], vals[::-1])
    plt.xlabel("Importance")
    plt.title(title)

    os.makedirs(FIG_DIR, exist_ok=True)
    safe = title.lower().replace(" ", "_").replace("|", "").replace("__", "_")
    fig.savefig(os.path.join(FIG_DIR, f"{safe}.png"), dpi=200, bbox_inches="tight")
    return fig
