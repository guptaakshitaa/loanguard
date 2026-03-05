"""Model registry — loads and caches trained pipeline artifacts.

Provides a single shared ModelRegistry instance injected into
FastAPI routes via Depends(). Handles champion/challenger routing
and exposes metadata for /health and /drift endpoints.

Usage:
    from api.model_registry import get_registry
    registry = get_registry()
    pipeline = registry.champion
    proba = pipeline.predict_proba(X)[:, 1]
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction is attempted before the model is loaded."""
    pass


class ModelRegistry:
    """Loads, caches, and serves trained model artifacts.

    Attributes:
        champion: The primary production pipeline.
        challenger: Optional A/B challenger pipeline.
        champion_meta: Metadata dict from the .meta.json sidecar.
        champion_version: Version string extracted from metadata.
        champion_path: Path to the loaded .pkl file.
        load_time: Unix timestamp when model was loaded.
    """

    def __init__(self) -> None:
        self.champion: Any = None
        self.challenger: Any = None
        self.champion_meta: dict[str, Any] = {}
        self.champion_version: str = "unknown"
        self.champion_path: str = ""
        self.load_time: float = 0.0
        self._training_data_probs: Optional[np.ndarray] = None

    def load(self, model_dir: str | Path = "models") -> None:
        """Load champion (and optionally challenger) from model_dir.

        Searches for champion_latest.pkl first, then falls back to
        the most recently modified champion_*.pkl.

        Args:
            model_dir: Directory containing .pkl artifacts.

        Raises:
            FileNotFoundError: If no champion model is found.
        """
        model_dir = Path(model_dir)

        # Try canonical latest path first
        candidates = [model_dir / "champion_latest.pkl"]
        # Then any versioned champion
        candidates += sorted(
            model_dir.glob("champion_*.pkl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        loaded = False
        for path in candidates:
            if path.exists() and path.stem != "champion_latest":
                # Already tried latest above
                pass
            if not path.exists():
                continue
            try:
                with open(path, "rb") as fh:
                    self.champion = pickle.load(fh)
                self.champion_path = str(path)
                self.load_time = time.time()
                self._load_meta(path)
                logger.info("Champion loaded: %s (version=%s)", path.name, self.champion_version)
                loaded = True
                break
            except Exception as exc:
                logger.warning("Failed to load %s: %s", path, exc)
                continue

        if not loaded:
            raise FileNotFoundError(
                f"No champion model found in {model_dir}. "
                "Run `python ml/train.py` first."
            )

        # Try challenger (non-fatal)
        chall_candidates = sorted(
            model_dir.glob("challenger_*.pkl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if chall_candidates:
            try:
                with open(chall_candidates[0], "rb") as fh:
                    self.challenger = pickle.load(fh)
                logger.info("Challenger loaded: %s", chall_candidates[0].name)
            except Exception as exc:
                logger.warning("Challenger load failed (non-fatal): %s", exc)

    def _load_meta(self, pkl_path: Path) -> None:
        """Load sidecar .meta.json if present."""
        meta_path = pkl_path.with_suffix(".meta.json")
        if meta_path.exists():
            try:
                with open(meta_path) as fh:
                    self.champion_meta = json.load(fh)
                self.champion_version = self.champion_meta.get("version", "unknown")
            except Exception as exc:
                logger.warning("Could not read meta: %s", exc)
        else:
            self.champion_version = pkl_path.stem.replace("champion_", "")

    def predict_proba(
        self,
        X: pd.DataFrame,
        use_challenger: bool = False,
    ) -> np.ndarray:
        """Run prediction and return default probabilities.

        Args:
            X: Feature DataFrame.
            use_challenger: Route to challenger if True and loaded.

        Returns:
            1-D array of default probabilities.

        Raises:
            ModelNotLoadedError: If model has not been loaded.
        """
        model = self.challenger if (use_challenger and self.challenger) else self.champion
        if model is None:
            raise ModelNotLoadedError(
                "Model not loaded. Call registry.load() before serving predictions."
            )
        return model.predict_proba(X)[:, 1]

    def store_training_probs(self, probs: np.ndarray) -> None:
        """Store training-set probabilities for PSI drift computation."""
        self._training_data_probs = probs

    def training_probs(self) -> Optional[np.ndarray]:
        """Return stored training probabilities (may be None)."""
        return self._training_data_probs

    @property
    def is_loaded(self) -> bool:
        return self.champion is not None

    @property
    def uptime_seconds(self) -> float:
        if self.load_time == 0:
            return 0.0
        return time.time() - self.load_time

    def selected_features(self) -> list[str]:
        """Return the feature list from model metadata."""
        return self.champion_meta.get("selected_features", [])


@lru_cache(maxsize=1)
def get_registry() -> ModelRegistry:
    """Return the singleton ModelRegistry.

    Called once at startup; subsequent calls return the cached instance.
    """
    return ModelRegistry()
