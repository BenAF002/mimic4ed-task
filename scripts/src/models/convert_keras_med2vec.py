"""Convert the legacy Keras Med2Vec SavedModels to `.npz` weight files for PyTorch.

Needs TensorFlow, which is deliberately not a project dependency. Run it in a throwaway env:

    uv run --no-project --python 3.12 --with tensorflow-cpu python \
        scripts/src/models/convert_keras_med2vec.py data/embeddings/embedding_hospitalization_v10 data/models

Output: `<out_dir>/med2vec_<task>_<version>.npz`, loadable with `models.med2vec.Med2Vec.from_npz`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

LAYERS = ["icd_fc", "fc1", "fc2", "out"]  # Dense layers after the embedding, in Keras order


def convert(saved_model_dir: str | Path, out_dir: str | Path) -> Path:
    """Write the weights of a Keras `create_embedding_model` SavedModel to an `.npz` file."""
    import tensorflow as tf  # imported lazily: only available in the throwaway env

    src = Path(saved_model_dir)
    reader = tf.train.load_checkpoint(str(src / "variables" / "variables"))
    get = lambda name: reader.get_tensor(f"layer_with_weights-{name}/.ATTRIBUTES/VARIABLE_VALUE")

    weights = {"embedding.weight": get("0/embeddings")}
    for i, layer in enumerate(LAYERS, start=1):
        weights[f"{layer}.weight"] = get(f"{i}/kernel").T  # Keras (in, out) -> PyTorch (out, in)
        weights[f"{layer}.bias"] = get(f"{i}/bias")

    # embedding_<task>_<version> -> med2vec_<task>_<version>
    out = Path(out_dir) / src.name.replace("embedding_", "med2vec_", 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **weights)
    print(f"{src.name}: {{{', '.join(f'{k}: {v.shape}' for k, v in weights.items())}}} -> {out}.npz")
    return out


if __name__ == "__main__":
    out_dir = sys.argv[-1]
    for d in sys.argv[1:-1]:
        convert(d, out_dir)
