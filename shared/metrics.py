from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def extract_encoder_feature(hidden_cache: dict, layer_idx: int) -> np.ndarray:
    hidden = hidden_cache[("enc", layer_idx)]
    if hidden.ndim == 3:
        vector = hidden[0].mean(dim=0)
    elif hidden.ndim == 2:
        vector = hidden.mean(dim=0)
    else:
        raise RuntimeError(f"Unexpected encoder hidden shape: {tuple(hidden.shape)}")
    return vector.float().cpu().numpy()


def extract_decoder_feature(hidden_cache: dict, layer_idx: int) -> np.ndarray:
    hidden = hidden_cache[("dec", layer_idx)]
    if hidden.ndim == 3:
        vector = hidden[0, -1, :]
    elif hidden.ndim == 2:
        vector = hidden[-1, :]
    else:
        raise RuntimeError(f"Unexpected decoder hidden shape: {tuple(hidden.shape)}")
    return vector.float().cpu().numpy()


def train_logistic_probe(
    layer_features: list[list[np.ndarray]],
    labels: np.ndarray,
    num_slots: int,
    test_size: float,
    seed: int,
) -> tuple[list[float], list[dict]]:
    layer_accuracies: list[float] = []
    layer_stats: list[dict] = []

    for slot_idx in range(num_slots):
        features = np.stack(layer_features[slot_idx], axis=0)
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=test_size,
            random_state=seed,
            stratify=labels,
        )
        classifier = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            LogisticRegression(max_iter=2000, random_state=seed),
        )
        classifier.fit(x_train, y_train)
        accuracy = float(classifier.score(x_test, y_test))
        layer_accuracies.append(accuracy)
        layer_stats.append(
            {
                "slot": slot_idx,
                "accuracy": accuracy,
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
            }
        )

    return layer_accuracies, layer_stats


def get_yes_no_probabilities(logits: torch.Tensor, tokenizer) -> tuple[float, float]:
    vocabulary = tokenizer.get_vocab()
    yes_ids = [vocabulary[token] for token in ["yes", " yes", "Yes", " Yes"] if token in vocabulary]
    no_ids = [vocabulary[token] for token in ["no", " no", "No", " No"] if token in vocabulary]
    if not yes_ids:
        yes_ids = [tokenizer.encode(" yes", add_special_tokens=False)[-1]]
    if not no_ids:
        no_ids = [tokenizer.encode(" no", add_special_tokens=False)[-1]]

    probabilities = torch.softmax(logits, dim=-1)
    return probabilities[yes_ids].max().item(), probabilities[no_ids].max().item()
