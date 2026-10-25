from pathlib import Path


SEED = 42
N_SPLITS = 5

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submissions"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

TARGET = "SalePrice"
ID_COLUMN = "Id"

MODEL_NAME = "voting"

MODEL_NAMES = {"voting": "Equal Voting",
               "catboost": "CatBoost One-Hot",
               "svr": "RBF SVR",
               "dnn": "PyTorch DNN"}

SUBMISSION_PATHS = {"voting": SUBMISSION_DIR / "house_prices_equal_voting.csv",
                    "catboost": SUBMISSION_DIR / "house_prices_catboost_one_hot.csv",
                    "svr": SUBMISSION_DIR / "house_prices_rbf_svr.csv",
                    "dnn": SUBMISSION_DIR / "house_prices_pytorch_dnn.csv"}

# Use the parameters selected in the notebook
CATBOOST_PARAMS = {"iterations": 1200,
                   "learning_rate": 0.030023005574603863,
                   "depth": 5,
                   "l2_leaf_reg": 1.767354970433315,
                   "random_strength": 3.721911570978738,
                   "border_count": 128,
                   "loss_function": "RMSE",
                   "random_seed": SEED,
                   "verbose": 0,
                   "allow_writing_files": False,
                   "thread_count": 1}

SVR_PARAMS = {"kernel": "rbf",
              "C": 18.4625000498051,
              "epsilon": 0.004342457951180244,
              "gamma": 0.00042166943832873887}

LASSO_PARAMS = {"alpha": 0.0005098460288377917,
                "max_iter": 20000}

DNN_PARAMS = {"hidden_size_1": 256,
              "hidden_size_2": 64,
              "dropout_1": 0.4,
              "dropout_2": 0.0,
              "learning_rate": 0.00013280145178866016,
              "weight_decay": 0.0010078448876277442,
              "batch_size": 128,
              "epochs": 50,
              "activation": "LeakyReLU",
              "batch_norm": False,
              "optimizer": "Adam",
              "scheduler": "None"}