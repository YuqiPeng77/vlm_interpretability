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


def _make_logistic_pipeline(seed: int):
    return make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(max_iter=2000, random_state=seed),
    )


def train_logistic_probe_with_random_baseline(
    layer_features: list[list[np.ndarray]],
    labels: np.ndarray,
    num_slots: int,
    test_size: float,
    seed: int,
    random_repeats: int = 3,
) -> tuple[list[float], list[dict]]:
    layer_accuracies: list[float] = []
    layer_stats: list[dict] = []

    for slot_idx in range(num_slots):
        features = np.stack(layer_features[slot_idx], axis=0)
        indices = np.arange(features.shape[0])
        train_idx, test_idx, y_train, y_test = train_test_split(
            indices,
            labels,
            test_size=test_size,
            random_state=seed,
            stratify=labels,
        )
        x_train = features[train_idx]
        x_test = features[test_idx]

        classifier = _make_logistic_pipeline(seed)
        classifier.fit(x_train, y_train)
        accuracy = float(classifier.score(x_test, y_test))
        layer_accuracies.append(accuracy)

        random_runs: list[float] = []
        y_train_base = np.array(y_train, copy=True)
        y_test_base = np.array(y_test, copy=True)
        for repeat_idx in range(random_repeats):
            repeat_seed = seed + repeat_idx
            train_rng = np.random.default_rng(repeat_seed)
            test_rng = np.random.default_rng(repeat_seed + 10_000)
            shuffled_train = np.array(y_train_base, copy=True)
            shuffled_test = np.array(y_test_base, copy=True)
            train_rng.shuffle(shuffled_train)
            test_rng.shuffle(shuffled_test)
            random_classifier = _make_logistic_pipeline(repeat_seed)
            random_classifier.fit(x_train, shuffled_train)
            random_runs.append(float(random_classifier.score(x_test, shuffled_test)))

        random_mean = float(np.mean(random_runs))
        random_std = float(np.std(random_runs))
        layer_stats.append(
            {
                "slot": slot_idx,
                "accuracy": accuracy,
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "random_accuracy_runs": [float(value) for value in random_runs],
                "random_accuracy_mean": random_mean,
                "random_accuracy_std": random_std,
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
