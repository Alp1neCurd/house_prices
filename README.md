# House Prices Prediction

Regression models for Kaggle House Prices

Equal Voting combines CatBoost, RBF SVR and Lasso and was selected by cross-validation
RBF SVR achieved the best submitted public score

## Files

- `main.py` - evaluation, training and submission export
- `config.py` - paths, model selection and parameters
- `features.py` - cleaning and feature engineering
- `model.py` - preprocessing and model construction
- `validation.py` - five-fold cross-validation
- `dnn.py` - PyTorch regressor
- `house_prices_experiments.ipynb` - EDA, tuning, comparisons and error analysis

## Setup

Tested with Python 3.14.6

Download `train.csv` and `test.csv` from the [competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) and place them in `data/`

From the project directory:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

For DNN:

```powershell
py -m pip install -r requirements_dnn.txt
```

For the complete notebook environment:

```powershell
py -m pip install -r requirements_notebook.txt
```

On Linux and macOS:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Use `python` instead of `py` for subsequent commands on Linux and macOS

## Run

Run the model selected by `MODEL_NAME` in `config.py`, currently Voting:

```powershell
py main.py
```

Override the model:

```powershell
py main.py --model svr
py main.py --model catboost
py main.py --model dnn
```

Also export averaged predictions from five training folds:

```powershell
py main.py --five-fold
```

The script prints validation metrics and saves CSV files in `submissions/`
DNN uses CUDA when available and otherwise runs on CPU
Other experimental models and categorical embeddings are available in the notebook

## Method

Median imputation, scaling and one-hot encoding are fitted inside each training fold
The pipeline adds total area, bathrooms, house age, remodel age and outdoor area

Two large low-price houses are excluded from the main experiments
A separate check keeps them in validation to examine the effect of removing them from training
Implausible garage years are replaced with the house construction year

Validation uses shuffled five-fold CV and RMSE on `log1p(SalePrice)`
Five-fold submissions average predictions in log-price space before converting them back to prices

## Results

Lower scores are better
CV results come from the notebook, and public scores come from the submitted CSV files

| Model | CV RMSE | CV std | Public score |
|---|---:|---:|---:|
| Equal Voting | 0.1071 | 0.0081 | 0.12266 |
| Stacking Ridge | 0.1072 | 0.0084 | 0.12260 |
| Equal Voting with DNN | 0.1073 | 0.0085 | 0.12316 |
| CatBoost One-Hot | 0.1111 | 0.0083 | 0.12617 |
| **RBF SVR** | 0.1113 | 0.0098 | **0.12072** |
| Lasso | 0.1116 | 0.0066 | 0.13322 |
| ElasticNet | 0.1117 | 0.0068 | 0.13307 |
| CatBoost Native | 0.1127 | 0.0069 | 0.12353 |
| PyTorch DNN | 0.1128 | 0.0083 | 0.12920 |
| XGBoost | 0.1131 | 0.0057 | 0.12971 |
| LightGBM | 0.1136 | 0.0073 | 0.12723 |
| Ridge | 0.1138 | 0.0085 | 0.13124 |
| Gradient Boosting | 0.1154 | 0.0042 | 0.13330 |
| PyTorch DNN Embeddings | 0.1196 | 0.0070 | 0.13138 |
| Random Forest | 0.1306 | 0.0076 | 0.14092 |
| Bagging Decision Tree | 0.1307 | 0.0073 | 0.14033 |
| KNN | 0.1573 | 0.0111 | 0.16582 |
| Decision Tree | 0.1753 | 0.0122 | 0.18492 |
| Equal Voting, five-fold submission | - | - | 0.12184 |

Voting had the lowest mean CV RMSE, while SVR had the best public score
The public leaderboard is used as an external check rather than the model selection criterion

DNN tuning uses different seeds across folds, so its notebook tuning score may differ from the script's CV result
Embeddings did not improve the DNN with the transferred hyperparameters

## Limitations

- Tuning and model selection reuse the same folds
- Main CV excludes the two flagged houses and may understate errors on unusual properties
- The benefit of outlier removal may depend on which unusual houses appear in training and validation
- Small score differences do not establish a reliable ranking
- GPU training results may vary between runs