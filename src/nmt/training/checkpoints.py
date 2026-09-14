"""Model checkpoint saving and loading."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Saves model checkpoints and tracks the best one.

    Args:
        checkpoint_dir: Directory to save checkpoints.
        monitor: Metric name to monitor (e.g. 'val_bleu').
        mode: 'max' if higher is better, 'min' if lower is better.
        save_best_only: If True, only keep the best checkpoint.
        filename_prefix: Prefix for checkpoint filenames.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        monitor: str = "val_bleu",
        mode: str = "max",
        save_best_only: bool = True,
        filename_prefix: str = "checkpoint",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.filename_prefix = filename_prefix
        self.best_value: Optional[float] = None
        self.best_path: Optional[Path] = None

    def save(
        self,
        model: nn.Module,
        epoch: int,
        metrics: Dict[str, float],
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        """Save a checkpoint if it is the best so far (or always if save_best_only=False).

        Args:
            model: Model to save.
            epoch: Current epoch number.
            metrics: Dictionary of metric values (must include self.monitor key).
            extra: Optional extra data to store (optimizer state, etc.).

        Returns:
            Path to saved checkpoint, or None if not saved.
        """
        current = metrics.get(self.monitor)
        if current is None:
            logger.warning("CheckpointManager: '%s' not found in metrics.", self.monitor)
            return None

        is_better = self._is_better(current)
        if self.save_best_only and not is_better:
            return None

        self.best_value = current
        filename = self.checkpoint_dir / f"{self.filename_prefix}_epoch{epoch:03d}.pt"

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
        }
        if extra:
            state.update(extra)

        torch.save(state, filename)
        self.best_path = filename
        logger.info(
            "Checkpoint saved: %s (%s=%.4f)", filename, self.monitor, current
        )
        return filename

    def load_best(self, model: nn.Module) -> Dict:
        """Load the best checkpoint into a model.

        Args:
            model: Model to load weights into.

        Returns:
            Full checkpoint dictionary.
        """
        if self.best_path is None or not self.best_path.exists():
            raise FileNotFoundError("No checkpoint found.")
        return self.load(model, str(self.best_path))

    @staticmethod
    def load(model: nn.Module, path: str) -> Dict:
        """Load a checkpoint from a file path.

        Args:
            model: Model to load weights into.
            path: Path to the .pt checkpoint file.

        Returns:
            Full checkpoint dictionary.
        """
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state["model_state_dict"])
        logger.info("Checkpoint loaded from %s (epoch %d)", path, state.get("epoch", -1))
        return state

    def _is_better(self, current: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "max":
            return current > self.best_value
        return current < self.best_value
