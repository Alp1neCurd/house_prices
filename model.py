from catboost import CatBoostRegressor

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

from config import CATBOOST_PARAMS, DNN_PARAMS, LASSO_PARAMS, SEED, SVR_PARAMS
from features import make_feature_engineer


def make_preprocessor():
    """Build numeric and categorical preprocessing"""

    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),
                                 ("scaler", StandardScaler())])

    categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                                     ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    return ColumnTransformer([("numeric", numeric_pipeline, make_column_selector(dtype_include="number")),
                              ("categorical", categorical_pipeline, make_column_selector(dtype_exclude="number"))])


def make_single_model_pipeline(model):
    """Combine feature engineering, preprocessing, and a regressor"""

    return Pipeline([("features", make_feature_engineer()),
                     ("preprocessor", make_preprocessor()),
                     ("model", model)])


def make_model_pipeline(model_name="voting"):
    """Build a configured regression model or Voting ensemble"""

    if model_name == "voting":
        return VotingRegressor(estimators=[("catboost", make_model_pipeline("catboost")),
                                           ("svr", make_model_pipeline("svr")),
                                           ("lasso", make_single_model_pipeline(Lasso(**LASSO_PARAMS)))],n_jobs=1)

    if model_name == "catboost":
        model = CatBoostRegressor(**CATBOOST_PARAMS)

    elif model_name == "svr":
        model = SVR(**SVR_PARAMS)

    elif model_name == "dnn":
        try:
            from dnn import TorchDNNRegressor
        except ImportError as error:
            raise ImportError("DNN dependencies are not installed. " "Run: pip install -r requirements_dnn.txt") from error

        model = TorchDNNRegressor(parameters=DNN_PARAMS, random_state=SEED)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return make_single_model_pipeline(model)