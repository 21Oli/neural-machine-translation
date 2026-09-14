"""Main training loop."""

import logging
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from .callbacks import CallbackList, EarlyStopping
from .checkpoints import CheckpointManager

logger = logging.getLogger(__name__)


class Trainer:
    """Encapsulates the training and evaluation loop.

    Args:
        model: The seq2seq model to train.
        optimizer: PyTorch optimizer.
        criterion: Loss function (e.g. nn.CrossEntropyLoss).
        device: Torch device.
        clip_grad_norm: Max gradient norm for clipping (0 = no clipping).
        teacher_forcing_ratio: Ratio for teacher forcing during training.
        scheduler: Optional learning rate scheduler.
        checkpoint_manager: Optional CheckpointManager.
        callbacks: Optional list of Callback instances.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: torch.device,
        clip_grad_norm: float = 1.0,
        teacher_forcing_ratio: float = 0.5,
        scheduler: Optional[_LRScheduler] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        callbacks: Optional[List] = None,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.clip_grad_norm = clip_grad_norm
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.scheduler = scheduler
        self.checkpoint_manager = checkpoint_manager
        self.callbacks = CallbackList(callbacks or [])
        self.history: Dict[str, List[float]] = {}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        tgt_pad_idx: int = 0,
        eval_fn: Optional[Callable] = None,
    ) -> Dict[str, List[float]]:
        """Run the full training loop.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            epochs: Number of training epochs.
            tgt_pad_idx: Target padding index (ignored in loss).
            eval_fn: Optional callable(model, val_loader) → dict of metrics.

        Returns:
            Training history dictionary.
        """
        self.callbacks.on_train_begin()

        for epoch in range(1, epochs + 1):
            self.callbacks.on_epoch_begin(epoch)

            train_loss = self._train_epoch(train_loader, tgt_pad_idx)
            val_loss = self._eval_epoch(val_loader, tgt_pad_idx)

            logs: Dict = {"train_loss": train_loss, "val_loss": val_loss}

            if eval_fn is not None:
                extra_metrics = eval_fn(self.model, val_loader)
                logs.update(extra_metrics)

            if self.scheduler is not None:
                if hasattr(self.scheduler, "step"):
                    metric = logs.get("val_loss", val_loss)
                    try:
                        self.scheduler.step(metric)
                    except TypeError:
                        self.scheduler.step()

            if self.checkpoint_manager is not None:
                self.checkpoint_manager.save(self.model, epoch, logs)

            for k, v in logs.items():
                self.history.setdefault(k, []).append(v)

            logger.info(
                "Epoch %d/%d — train_loss: %.4f  val_loss: %.4f",
                epoch, epochs, train_loss, val_loss,
            )

            self.callbacks.on_epoch_end(epoch, logs)

            # Check early stopping
            for cb in self.callbacks.callbacks:
                if isinstance(cb, EarlyStopping) and cb.stop_training:
                    logger.info("Early stopping triggered at epoch %d.", epoch)
                    self.callbacks.on_train_end(logs)
                    return self.history

        self.callbacks.on_train_end()
        return self.history

    def _train_epoch(self, loader: DataLoader, tgt_pad_idx: int) -> float:
        self.model.train()
        total_loss = 0.0

        for batch_idx, (src, tgt) in enumerate(loader):
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            self.optimizer.zero_grad()

            # Support both seq2seq (output) and attention_seq2seq (output, attentions)
            result = self.model(src, tgt, self.teacher_forcing_ratio)
            output = result[0] if isinstance(result, tuple) else result

            # output: (batch, tgt_len-1, vocab_size) → (batch*(tgt_len-1), vocab_size)
            output_flat = output.reshape(-1, output.shape[-1])
            tgt_flat = tgt[:, 1:].reshape(-1)

            loss = self.criterion(output_flat, tgt_flat)
            loss.backward()

            if self.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

            self.optimizer.step()
            total_loss += loss.item()

            self.callbacks.on_batch_end(batch_idx, {"loss": loss.item()})

        return total_loss / len(loader)

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader, tgt_pad_idx: int) -> float:
        self.model.eval()
        total_loss = 0.0

        for src, tgt in loader:
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            result = self.model(src, tgt, teacher_forcing_ratio=0.0)
            output = result[0] if isinstance(result, tuple) else result

            output_flat = output.reshape(-1, output.shape[-1])
            tgt_flat = tgt[:, 1:].reshape(-1)

            loss = self.criterion(output_flat, tgt_flat)
            total_loss += loss.item()

        return total_loss / len(loader)
