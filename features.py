from sklearn.preprocessing import FunctionTransformer

from config import TARGET


def remove_training_outliers(data):
    """Remove the two flagged large low-price houses"""

    outlier_mask = ((data["GrLivArea"] > 4000) & (data[TARGET] < 300000))

    return data.loc[~outlier_mask].copy()


def add_features(data):
    """Clean housing features and add aggregate measurements"""

    data = data.copy()

    # Use the house year as a fallback for implausible garage years
    invalid_garage_year = data["GarageYrBlt"] > data["YrSold"] + 1

    data.loc[invalid_garage_year, "GarageYrBlt"] = data.loc[invalid_garage_year, "YearBuilt"]
    data["TotalSF"] = data[["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]].sum(axis=1)
    data["TotalBathrooms"] = (data["FullBath"].fillna(0) + 0.5 * data["HalfBath"].fillna(0) + data["BsmtFullBath"].fillna(0) + 0.5 * data["BsmtHalfBath"].fillna(0))
    data["HouseAge"] = (data["YrSold"] - data["YearBuilt"]).clip(lower=0)
    data["RemodelAge"] = (data["YrSold"] - data["YearRemodAdd"]).clip(lower=0)
    data["TotalOutdoorSF"] = data[["WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]].sum(axis=1)
    data["MSSubClass"] = data["MSSubClass"].astype(str)

    return data


def make_feature_engineer():
    """Build the housing feature transformer"""

    return FunctionTransformer(add_features, validate=False)