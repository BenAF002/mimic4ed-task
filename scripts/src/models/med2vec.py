"""PyTorch port of the Med2Vec-style ICD-embedding model from the original Keras benchmark.

ICD codes are embedded and summed per stay, passed through a dense layer, concatenated with the
demographic / triage features and fed to a small MLP.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch import nn


class Med2Vec(nn.Module):
    """ICD embedding-sum + feature MLP producing a probability per stay.

    Inputs are `demo` (batch, demographic_size) floats and `icd` (batch, max_len) ints holding
    1-based code indices, with 0 as padding (see `pad_icd`).

    The embedding deliberately has no `padding_idx`: the Keras original summed the padding row
    (index 0) as well (`mask_zero` is ignored by `keras.backend.sum`), so pretrained weights only
    reproduce Keras outputs if padding is included.
    """

    def __init__(self, vocabulary: int, demographic_size: int = 64, embedding_dim: int = 1024):
        super().__init__()
        self.vocabulary = vocabulary
        self.embedding = nn.Embedding(vocabulary + 1, embedding_dim)  # +1 for the padding index
        self.icd_fc = nn.Linear(embedding_dim, 256)
        self.fc1 = nn.Linear(256 + demographic_size, 128)
        self.fc2 = nn.Linear(128, 50)
        self.out = nn.Linear(50, 1)
        self._init_like_keras()

    def _init_like_keras(self) -> None:
        """Keras defaults: embeddings ~ U(-0.05, 0.05), Glorot-uniform kernels, zero biases.

        PyTorch's N(0, 1) embedding init makes the summed embedding far too large for stays with
        many codes, which badly hurts training.
        """
        nn.init.uniform_(self.embedding.weight, -0.05, 0.05)
        for layer in (self.icd_fc, self.fc1, self.fc2, self.out):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, demo: torch.Tensor, icd: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.icd_fc(self.embedding(icd).sum(dim=1)))
        h = torch.relu(self.fc1(torch.cat([h, demo], dim=1)))
        return torch.sigmoid(self.out(torch.relu(self.fc2(h)))).squeeze(1)

    @classmethod
    def from_npz(cls, path: str | Path) -> "Med2Vec":
        """Load weights converted by `convert_keras_med2vec.py`."""
        w = np.load(path)
        demographic_size = w["fc1.weight"].shape[1] - w["icd_fc.weight"].shape[0]
        model = cls(w["embedding.weight"].shape[0] - 1, demographic_size, w["embedding.weight"].shape[1])
        model.load_state_dict({k: torch.from_numpy(w[k]) for k in model.state_dict()})
        return model.eval()


def pad_icd(icd_lists: list[list[int]]) -> torch.Tensor:
    """Pad 0-based code index lists into a (batch, max_len) tensor of 1-based indices (0 = pad)."""
    max_len = max(1, max((len(x) for x in icd_lists), default=1))
    out = np.zeros((len(icd_lists), max_len), dtype=np.int64)
    for i, codes in enumerate(icd_lists):
        out[i, :len(codes)] = np.asarray(codes, dtype=np.int64) + 1
    return torch.from_numpy(out)


def batches(X: np.ndarray, icd_lists: list[list[int]], y: np.ndarray | None = None,
            batch_size: int = 200, order: np.ndarray | None = None) -> Iterator[tuple]:
    """Yield `(demo, icd[, y])` batches, each padded to its own longest ICD list.

    Because padding contributes to the embedding sum, outputs depend slightly on batch makeup.
    Use sequential order and `batch_size=200` to match the original benchmark.
    """
    order = np.arange(len(X)) if order is None else order
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        batch = (torch.as_tensor(X[idx], dtype=torch.float32), pad_icd([icd_lists[i] for i in idx]))
        yield batch if y is None else (*batch, torch.as_tensor(y[idx], dtype=torch.float32))


@torch.no_grad()
def predict(model: Med2Vec, X: np.ndarray, icd_lists: list[list[int]], batch_size: int = 200) -> np.ndarray:
    """Predicted probabilities for every row, in order."""
    model.eval()
    device = next(model.parameters()).device
    return np.concatenate([model(d.to(device), i.to(device)).cpu().numpy()
                           for d, i in batches(X, icd_lists, batch_size=batch_size)])
