from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def make_logistic_regression(seed: int):
    return LogisticRegression(
        max_iter=5000,
        random_state=seed,
        solver="lbfgs"
    )

def make_random_forest(seed: int):
    return RandomForestClassifier(
        n_estimators=300,
        random_state=seed,
        n_jobs=-1
    )
