"""Training callbacks for monitoring and early stopping."""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Callback:
    """Base class for training callbacks.

    Subclass and override the relevant hook methods.
    """

    def on_train_begin(self, logs: Optional[Dict] = None) -> None:
        pass

    def on_train_end(self, logs: Optional[Dict] = None) -> None:
        pass

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None) -> None:
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        pass

    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None) -> None:
        pass

    def on_batch_end(self, batch: int, logs: Optional[Dict] = None) -> None:
        pass


class CallbackList:
    """Container that calls each callback in order.

    Args:
        callbacks: List of Callback instances.
    """

    def __init__(self, callbacks: Optional[List[Callback]] = None) -> None:
        self.callbacks = callbacks or []

    def append(self, callback: Callback) -> None:
        self.callbacks.append(callback)

    def on_train_begin(self, logs: Optional[Dict] = None) -> None:
        for cb in self.callbacks:
            cb.on_train_begin(logs)

    def on_train_end(self, logs: Optional[Dict] = None) -> None:
        for cb in self.callbacks:
            cb.on_train_end(logs)

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None) -> None:
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None) -> None:
        for cb in self.callbacks:
            cb.on_batch_begin(batch, logs)

    def on_batch_end(self, batch: int, logs: Optional[Dict] = None) -> None:
        for cb in self.callbacks:
            cb.on_batch_end(batch, logs)


class EarlyStopping(Callback):
    """Stop training when a monitored metric stops improving.

    Args:
        monitor: Metric name to watch (e.g. 'val_loss').
        patience: Number of epochs with no improvement before stopping.
        mode: 'min' for loss-style metrics, 'max' for score-style metrics.
        min_delta: Minimum change to count as improvement.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 5,
        mode: str = "min",
        min_delta: float = 1e-4,
    ) -> None:
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best: Optional[float] = None
        self.wait = 0
        self.stopped_epoch = 0
        self.stop_training = False

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            logger.warning("EarlyStopping: metric '%s' not found in logs.", self.monitor)
            return

        if self.best is None:
            self.best = current
            return

        improved = (
            current < self.best - self.min_delta
            if self.mode == "min"
            else current > self.best + self.min_delta
        )

        if improved:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True
                logger.info(
                    "EarlyStopping: no improvement in '%s' for %d epochs. Stopping at epoch %d.",
                    self.monitor,
                    self.patience,
                    epoch,
                )
