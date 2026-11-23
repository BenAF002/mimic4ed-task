"""
Data preparation and management utilities for model benchmarking notebooks.
Includes boilerplate DataSet and DataLoader classes expected by PyTorch.
"""

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from dataclasses import dataclass

from models.config import Config


class DataPartition:
    """Thin wrapper to hold partitions of dataset. `X` and `y` are NumPy arrays."""
    def __init__(self, 
                 df: pl.DataFrame | pl.LazyFrame,
                 target: str,
                 config: Config | None = None,
                 features: list[str] | None = None):
        self.df = df
        self.config = Config() if config is None else config
        self.features = features if features is not None else self.config.features_list
        self.target = target

        # polars frames narrowed down to features and target
        self.X_pl = df.select(self.features).collect() if isinstance(df, pl.LazyFrame) else df.select(self.features)
        self.y_pl = df.select(self.target).collect() if isinstance(df, pl.LazyFrame) else df.select(self.target)

        # create numpy arrays
        self.X = self.X_pl.to_numpy()
        self.y = self.y_pl.to_numpy().ravel()
