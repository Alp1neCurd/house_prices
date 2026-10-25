import argparse

import numpy as np
import pandas as pd

from sklearn.base import clone

from config import ID_COLUMN, MODEL_NAME, MODEL_NAMES, SUBMISSION_PATHS, TARGET, TEST_PATH, TRAIN_PATH
from features import remove_training_outliers
from model import make_model_pipeline
from validation import evaluate_model, make_cv


def parse_args():
    """Read model and submission options"""

    parser = argparse.ArgumentParser(description="Train and evaluate a House Prices regression model")
    parser.add_argument("--model", choices=MODEL_NAMES, default=MODEL_NAME, help="Model to run")
    parser.add_argument("--five-fold", action="store_true", help="Also export averaged five-fold predictions")

    return parser.parse_args()


def load_data():
    """Load House Prices data and remove flagged training outliers"""

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    train_df = remove_training_outliers(train_df)

    return train_df, test_df


def save_submission(test_ids, log_predictions, submission_path):
    """Convert log predictions to prices and save a submission"""

    prices = np.maximum(np.expm1(log_predictions), 0)

    if len(prices) != len(test_ids) or not np.isfinite(prices).all():
        raise ValueError("Submission predictions are invalid")

    submission = pd.DataFrame({ID_COLUMN: test_ids,
                               TARGET: prices})

    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_path, index=False)


def predict_full_fit(model, X, y, test_X):
    """Fit the model on all retained training houses and predict"""

    fitted_model = clone(model)
    fitted_model.fit(X, y)

    return fitted_model.predict(test_X)


def predict_five_fold(model, X, y, test_X):
    """Average test predictions from models fitted on five folds"""

    fold_predictions = []

    for train_index, _ in make_cv().split(X):
        fold_model = clone(model)
        fold_model.fit(X.iloc[train_index], y.iloc[train_index])

        fold_predictions.append(fold_model.predict(test_X))

    # Average in log-price space, matching the notebook
    return np.mean(fold_predictions, axis=0)


def main():
    """Evaluate a model and export full-fit or additional fold predictions"""

    args = parse_args()
    train_df, test_df = load_data()

    X = train_df.drop(columns=[ID_COLUMN, TARGET])
    y = np.log1p(train_df[TARGET])

    test_ids = test_df[ID_COLUMN]
    test_X = test_df.drop(columns=ID_COLUMN)

    model = make_model_pipeline(model_name=args.model)

    # Run GPU training sequentially across validation folds
    n_jobs = 1 if args.model == "dnn" else -1

    print("Model:", MODEL_NAMES[args.model])

    metrics = evaluate_model(model, X, y, n_jobs=n_jobs)

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    submission_path = SUBMISSION_PATHS[args.model]

    log_predictions = predict_full_fit(model, X, y, test_X)
    save_submission(test_ids, log_predictions, submission_path)

    print("Submission saved to:", submission_path)

    if args.five_fold:
        fold_predictions = predict_five_fold(model, X, y, test_X)
        five_fold_path = submission_path.with_name(f"{submission_path.stem}_5fold.csv")
        save_submission(test_ids, fold_predictions, five_fold_path)
        print("Five-fold submission saved to:", five_fold_path)


if __name__ == "__main__":
    main()