import numpy as np
import torch
import torch.nn as nn

from sklearn.base import BaseEstimator, RegressorMixin
from torch.utils.data import DataLoader, TensorDataset


def build_dnn(parameters, input_size):
    """Build a neural network with two hidden layers for regression"""

    activation_classes = {"ReLU": nn.ReLU,
                          "LeakyReLU": nn.LeakyReLU,
                          "ELU": nn.ELU}

    activation_class = activation_classes[parameters["activation"]]

    layers = [nn.Linear(input_size, parameters["hidden_size_1"]), activation_class()]

    if parameters["batch_norm"]:
        layers.append(nn.BatchNorm1d(parameters["hidden_size_1"]))

    layers.extend([nn.Dropout(parameters["dropout_1"]),
                   nn.Linear(parameters["hidden_size_1"], parameters["hidden_size_2"]),
                   activation_class(),
                   nn.Dropout(parameters["dropout_2"]),
                   nn.Linear(parameters["hidden_size_2"], 1)])

    return nn.Sequential(*layers)


def train_torch_model(parameters, features, targets, seed, model_factory):
    """Train a PyTorch model with a fixed random seed"""

    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_tensor = torch.tensor(np.asarray(features), dtype=torch.float32)

    target_tensor = torch.tensor(np.asarray(targets), dtype=torch.float32).reshape(-1, 1)

    generator = torch.Generator().manual_seed(seed)

    # Drop incomplete batches when BatchNorm is enabled
    train_loader = DataLoader(TensorDataset(feature_tensor, target_tensor), batch_size=parameters["batch_size"], shuffle=True, generator=generator, drop_last=parameters["batch_norm"])

    model = model_factory(feature_tensor.shape[1]).to(device)
    criterion = nn.MSELoss()

    optimizer_class = (torch.optim.AdamW if parameters["optimizer"] == "AdamW" else torch.optim.Adam)
    optimizer = optimizer_class(model.parameters(), lr=parameters["learning_rate"], weight_decay=parameters["weight_decay"])

    scheduler = None

    if parameters["scheduler"] == "Cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=parameters["epochs"], eta_min=1e-5)

    for _ in range(parameters["epochs"]):
        model.train()

        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()

            loss = criterion(model(batch_features), batch_targets)
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

    model.eval()

    return model, device


def train_dnn(parameters, features, targets, seed):
    """Train the one-hot DNN"""

    return train_torch_model(parameters, features, targets, seed, lambda input_size: build_dnn(parameters, input_size))


class TorchDNNRegressor(RegressorMixin, BaseEstimator):
    """Use the PyTorch DNN through the sklearn estimator interface"""

    def __init__(self, parameters=None, random_state=42):
        self.parameters = parameters
        self.random_state = random_state

    def fit(self, X, y):
        """Train the DNN regressor on the input data"""

        if self.parameters is None:
            raise ValueError("DNN parameters are required")

        features = np.asarray(X)
        targets = np.asarray(y, dtype=np.float32)

        # Learn target scaling from this training split only
        self.target_mean_ = float(targets.mean())
        self.target_std_ = float(targets.std())
        scaled_targets = (targets - self.target_mean_) / self.target_std_

        self.model_, self.device_ = train_dnn(self.parameters, features, scaled_targets, self.random_state)

        self.n_features_in_ = features.shape[1]

        return self

    def predict(self, X):
        """Return predicted log-transformed prices"""

        features = torch.tensor(np.asarray(X), dtype=torch.float32, device=self.device_)

        with torch.no_grad():
            scaled_predictions = self.model_(features).cpu().numpy().ravel()

        # Restore the original log-price scale
        return scaled_predictions * self.target_std_ + self.target_mean_