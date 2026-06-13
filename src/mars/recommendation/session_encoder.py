from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:  # pragma: no cover - optional local dependency; Docker runtime installs torch.
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


@dataclass(slots=True)
class SessionEncodingResult:
    vector: list[float]
    encoder_type: str
    sequence_length: int
    source_products: list[str]


@dataclass(slots=True)
class GRUSessionEncoder:
    """Encode short-term session interest from recent item embeddings with a GRU."""

    embedding_dim: int
    seed: int = 42
    max_sequence_length: int = 20
    model_payload: dict[str, Any] | None = None
    _gru: Any = field(default=None, init=False, repr=False)
    _trained: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if torch is None or nn is None:
            return
        torch.manual_seed(self.seed)
        self._gru = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.embedding_dim,
            num_layers=1,
            batch_first=True,
        )
        self._trained = self._load_payload()
        if not self._trained:
            _initialise_gru_module(self._gru, self.seed)
        self._gru.eval()

    def encode(
        self,
        recent_product_ids: list[str],
        item_embeddings: dict[str, list[float]],
    ) -> SessionEncodingResult | None:
        sequence: list[list[float]] = []
        source_products: list[str] = []
        # Session contexts expose newest-first IDs; GRU input must be chronological.
        chronological_ids = list(reversed(recent_product_ids[: self.max_sequence_length]))
        for product_id in chronological_ids:
            vector = item_embeddings.get(str(product_id))
            if not vector:
                continue
            sequence.append([float(value) for value in vector[: self.embedding_dim]])
            source_products.append(str(product_id))
        if not sequence:
            return None

        matrix = np.asarray(sequence, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.embedding_dim:
            return None

        if self._gru is not None and torch is not None:
            with torch.inference_mode():
                tensor = torch.from_numpy(matrix).unsqueeze(0)
                _, hidden = self._gru(tensor)
                vector = hidden[-1, 0].detach().cpu().numpy()
            return SessionEncodingResult(
                vector=_l2_normalize(vector).tolist(),
                encoder_type="gru_trained" if self._trained else "gru_untrained",
                sequence_length=len(sequence),
                source_products=source_products,
            )

        weights = np.linspace(0.4, 1.0, num=len(sequence), dtype=np.float32)
        pooled = (matrix * weights[:, None]).sum(axis=0) / max(float(weights.sum()), 1e-6)
        return SessionEncodingResult(
            vector=_l2_normalize(pooled).tolist(),
            encoder_type="pooled_fallback",
            sequence_length=len(sequence),
            source_products=source_products,
        )

    def _load_payload(self) -> bool:
        if self._gru is None or torch is None or not self.model_payload:
            return False
        if self.model_payload.get("model_type") != "torch_gru_session_encoder":
            return False
        try:
            state_dict = {
                key: torch.tensor(value, dtype=torch.float32)
                for key, value in dict(self.model_payload.get("state_dict", {})).items()
            }
            self._gru.load_state_dict(state_dict)
            return True
        except Exception:
            return False


def fit_gru_session_encoder(
    *,
    events: list[dict[str, Any]],
    item_embeddings: dict[str, list[float]],
    embedding_dim: int,
    seed: int,
    max_sequence_length: int = 20,
    max_samples: int = 4_000,
) -> dict[str, Any] | None:
    if torch is None or nn is None or not events or not item_embeddings:
        return None
    _stabilise_windows_torch_training()

    sessions: dict[str, list[str]] = {}
    samples: list[tuple[list[str], str]] = []
    for event in events:
        metadata = event.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        event_type = str(event.get("event_type", ""))
        product_id = str(event.get("product_id", ""))
        if (
            event_type not in {"view", "cart", "purchase"}
            or str(metadata.get("event_role", "")) == "exposure"
            or product_id not in item_embeddings
        ):
            continue
        session_id = str(event.get("session_id") or event.get("user_id") or "global")
        sequence = sessions.setdefault(session_id, [])
        if sequence:
            samples.append((sequence[-max_sequence_length:], product_id))
            if len(samples) >= max_samples:
                break
        sequence.append(product_id)

    if not samples:
        return None

    torch.manual_seed(seed)
    model = nn.GRU(
        input_size=embedding_dim,
        hidden_size=embedding_dim,
        num_layers=1,
        batch_first=True,
    )
    _initialise_gru_module(model, seed)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    batch_size = min(128, max(16, len(samples)))
    last_loss = 0.0
    model.train()
    for _epoch in range(3):
        order = torch.randperm(len(samples), generator=generator).tolist()
        for start in range(0, len(samples), batch_size):
            batch = [samples[index] for index in order[start : start + batch_size]]
            sequence_tensor = torch.zeros(
                (len(batch), max_sequence_length, embedding_dim),
                dtype=torch.float32,
            )
            targets: list[list[float]] = []
            for row, (product_ids, target_id) in enumerate(batch):
                vectors = [
                    item_embeddings[product_id][:embedding_dim]
                    for product_id in product_ids
                    if product_id in item_embeddings
                ][-max_sequence_length:]
                if vectors:
                    sequence_tensor[row, -len(vectors) :] = torch.tensor(
                        vectors, dtype=torch.float32
                    )
                targets.append(item_embeddings[target_id][:embedding_dim])
            target_tensor = torch.tensor(targets, dtype=torch.float32)
            _, hidden = model(sequence_tensor)
            prediction = torch.nn.functional.normalize(hidden[-1], dim=-1)
            target_tensor = torch.nn.functional.normalize(target_tensor, dim=-1)
            loss = (1.0 - torch.nn.functional.cosine_similarity(prediction, target_tensor)).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            last_loss = float(loss.detach().cpu().item())
    model.eval()
    return {
        "model_type": "torch_gru_session_encoder",
        "embedding_dim": embedding_dim,
        "max_sequence_length": max_sequence_length,
        "state_dict": {
            key: value.detach().cpu().tolist() for key, value in model.state_dict().items()
        },
        "trained_samples": len(samples),
        "objective": "next-item embedding cosine similarity",
        "epochs": 3,
        "final_loss": last_loss,
        "seed": seed,
    }


def combine_user_and_session_vectors(
    user_vector: list[float],
    session_vector: list[float],
    *,
    long_term_weight: float,
    session_weight: float,
) -> list[float]:
    user = np.asarray(user_vector, dtype=np.float32)
    session = np.asarray(session_vector, dtype=np.float32)
    if user.size == 0:
        return _l2_normalize(session).tolist()
    if session.size == 0:
        return _l2_normalize(user).tolist()
    if user.shape != session.shape:
        return _l2_normalize(user).tolist()
    long_term_weight = max(float(long_term_weight), 0.0)
    session_weight = max(float(session_weight), 0.0)
    total = long_term_weight + session_weight
    if total <= 1e-9:
        long_term_weight, session_weight, total = 0.7, 0.3, 1.0
    combined = ((long_term_weight / total) * user) + ((session_weight / total) * session)
    return _l2_normalize(combined).tolist()


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def _stabilise_windows_torch_training() -> None:
    if torch is None or not sys.platform.startswith("win"):
        return
    try:
        if torch.get_num_threads() > 1:
            torch.set_num_threads(1)
    except Exception:
        pass
    try:
        if torch.get_num_interop_threads() > 1:
            torch.set_num_interop_threads(1)
    except Exception:
        pass


def _initialise_gru_module(model: Any, seed: int) -> None:
    if model is None or torch is None or nn is None:
        return
    generator = torch.Generator().manual_seed(seed)
    for name, parameter in model.named_parameters():
        if "weight" in name:
            nn.init.xavier_uniform_(parameter, generator=generator)
        elif "bias" in name:
            nn.init.zeros_(parameter)
