from sklearn.model_selection import KFold, cross_validate

from config import N_SPLITS, SEED


def make_cv():
    """Build shuffled K-fold cross-validation"""

    return KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)


def evaluate_model(model, X, y, n_jobs=-1):
    """Evaluate a regressor using RMSE on log-transformed prices"""

    scores = cross_validate(model, X, y, cv=make_cv(), scoring="neg_root_mean_squared_error", return_train_score=True, n_jobs=n_jobs)

    return {"Train RMSE": -scores["train_score"].mean(),
            "CV RMSE": -scores["test_score"].mean(),
            "CV std": scores["test_score"].std()}