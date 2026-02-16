from dataclasses import dataclass
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score

@dataclass
class RocData:
    fpr: list
    tpr: list
    auc: float
    label: str

def evaluate_classifier(model, X_test, y_test, name="model"):
    y_pred = model.predict(X_test)
    metrics = {"model": name, "accuracy": float(accuracy_score(y_test, y_pred))}

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
        metrics["auc"] = float(roc_auc_score(y_test, y_score))
    else:
        metrics["auc"] = None

    return metrics

def roc_curve_data(model, X_test, y_test, label="model"):
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    auc = roc_auc_score(y_test, y_score)
    return RocData(fpr=fpr.tolist(), tpr=tpr.tolist(), auc=float(auc), label=label)

def cross_validate_model(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    return {
        "cv_mean_accuracy": float(np.mean(scores)),
        "cv_std_accuracy": float(np.std(scores)),
        "fold_scores": scores.tolist()
    }
