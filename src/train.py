"""
Training Pipeline for OrdinalChess

Implements the training loop with:
1. Multi-head loss optimization (value, policy, ordinal)
2. Curriculum learning: start with finite positions, gradually add transfinite
3. Evaluation metrics: ordinal accuracy, Kendall's tau for ordering
4. Checkpointing and logging
"""

from __future__ import annotations
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import json

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torch.cuda.amp import GradScaler, autocast
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import numpy as np
from scipy import stats as scipy_stats

from .transformer import OrdinalChessTransformer, TransformerConfig, OrdinalLoss
from .ordinals import OrdinalBucketizer, OrdinalTier
from .data import OrdinalChessDataset, create_dataloaders, analyze_dataset


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # Model
    model_size: str = "small"  # small, medium, large

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100000
    gradient_accumulation: int = 1

    # Data
    n_train_samples: int = 100000
    n_val_samples: int = 10000
    num_workers: int = 4

    # Curriculum (gradually introduce transfinite positions)
    curriculum_enabled: bool = True
    curriculum_start_step: int = 5000   # Start adding transfinite after N steps
    curriculum_end_step: int = 30000    # Full transfinite ratio by step N
    transfinite_final_ratio: float = 0.3  # Final ratio of transfinite samples

    # Logging and checkpoints
    log_interval: int = 100
    eval_interval: int = 1000
    checkpoint_interval: int = 5000
    checkpoint_dir: str = "checkpoints"

    # Hardware
    device: str = "cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu"
    use_amp: bool = True  # Automatic mixed precision

    # Loss weights
    value_weight: float = 1.0
    policy_weight: float = 0.5
    ordinal_weight: float = 2.0  # Higher weight for ordinal prediction


class Trainer:
    """Main training class for OrdinalChess."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Initialize model
        model_configs = {
            "small": TransformerConfig.small(),
            "medium": TransformerConfig.medium(),
            "large": TransformerConfig.large(),
        }
        model_config = model_configs[config.model_size]

        self.model = OrdinalChessTransformer(model_config)
        self.model.to(self.device)

        # Initialize optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_steps,
            eta_min=config.learning_rate / 10,
        )

        # Loss function
        self.loss_fn = OrdinalLoss(
            self.model.bucketizer,
            value_weight=config.value_weight,
            policy_weight=config.policy_weight,
            ordinal_weight=config.ordinal_weight,
        )

        # Gradient scaler for mixed precision
        self.scaler = GradScaler() if config.use_amp else None

        # Metrics tracking
        self.metrics_history: List[Dict] = []
        self.step = 0

        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def prepare_batch(self, batch: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Move batch to device and prepare for model input."""
        return {
            'tokens': torch.tensor(batch['tokens']).to(self.device),
            'rel_positions': torch.tensor(batch['position_ids']).to(self.device),
            'turn': torch.tensor(batch['turn']).squeeze(-1).to(self.device),
            'targets': {
                'ordinal': torch.tensor(batch['ordinal_target']).squeeze(-1).to(self.device),
                'value': torch.tensor(batch['value_target']).squeeze(-1).to(self.device),
            }
        }

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute single training step."""
        self.model.train()

        with autocast(enabled=self.config.use_amp):
            # Forward pass
            outputs = self.model(
                batch['tokens'],
                batch['rel_positions'],
                batch['turn'],
            )

            # Compute loss
            losses = self.loss_fn(outputs, batch['targets'])

        # Backward pass
        if self.scaler:
            self.scaler.scale(losses['total']).backward()

            if (self.step + 1) % self.config.gradient_accumulation == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            losses['total'].backward()

            if (self.step + 1) % self.config.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

        self.scheduler.step()

        return {k: v.item() for k, v in losses.items()}

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation set."""
        self.model.eval()

        total_losses = {}
        all_ordinal_preds = []
        all_ordinal_targets = []
        all_value_preds = []
        all_value_targets = []

        for batch in val_loader:
            batch = self.prepare_batch(batch)

            outputs = self.model(
                batch['tokens'],
                batch['rel_positions'],
                batch['turn'],
            )

            losses = self.loss_fn(outputs, batch['targets'])

            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0) + v.item()

            # Collect predictions for metrics
            all_ordinal_preds.extend(outputs['ordinal_predicted_bucket'].cpu().numpy())
            all_ordinal_targets.extend(batch['targets']['ordinal'].cpu().numpy())
            all_value_preds.extend(outputs['value'].argmax(dim=1).cpu().numpy())
            all_value_targets.extend(batch['targets']['value'].cpu().numpy())

        n_batches = len(val_loader)
        avg_losses = {k: v / n_batches for k, v in total_losses.items()}

        # Compute additional metrics
        ordinal_preds = np.array(all_ordinal_preds)
        ordinal_targets = np.array(all_ordinal_targets)
        value_preds = np.array(all_value_preds)
        value_targets = np.array(all_value_targets)

        # Ordinal accuracy
        avg_losses['ordinal_accuracy'] = (ordinal_preds == ordinal_targets).mean()

        # Value accuracy
        avg_losses['value_accuracy'] = (value_preds == value_targets).mean()

        # Kendall's tau for ordinal ordering
        if len(np.unique(ordinal_preds)) > 1 and len(np.unique(ordinal_targets)) > 1:
            tau, _ = scipy_stats.kendalltau(ordinal_preds, ordinal_targets)
            avg_losses['ordinal_kendall_tau'] = tau
        else:
            avg_losses['ordinal_kendall_tau'] = 0.0

        # Tier-wise accuracy
        bucketizer = self.model.bucketizer
        for tier in OrdinalTier:
            tier_mask = self._get_tier_mask(ordinal_targets, bucketizer, tier)
            if tier_mask.sum() > 0:
                tier_acc = (ordinal_preds[tier_mask] == ordinal_targets[tier_mask]).mean()
                avg_losses[f'accuracy_{tier.name.lower()}'] = tier_acc

        return avg_losses

    def _get_tier_mask(
        self,
        bucket_indices: np.ndarray,
        bucketizer: OrdinalBucketizer,
        tier: OrdinalTier
    ) -> np.ndarray:
        """Create mask for samples belonging to a specific ordinal tier."""
        mask = np.zeros(len(bucket_indices), dtype=bool)

        for i, bucket in enumerate(bucket_indices):
            ordinal = bucketizer.from_bucket(bucket)
            if ordinal.tier() == tier:
                mask[i] = True

        return mask

    def save_checkpoint(self, path: Optional[str] = None):
        """Save model checkpoint."""
        if path is None:
            path = os.path.join(
                self.config.checkpoint_dir,
                f"checkpoint_step_{self.step}.pt"
            )

        checkpoint = {
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'metrics_history': self.metrics_history,
        }

        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.step = checkpoint['step']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.metrics_history = checkpoint.get('metrics_history', [])

        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(f"Loaded checkpoint from {path} (step {self.step})")

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop."""
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())/1e6:.2f}M")

        start_time = time.time()
        running_losses = {}

        while self.step < self.config.max_steps:
            for batch in train_loader:
                batch = self.prepare_batch(batch)

                # Training step
                step_losses = self.train_step(batch)

                # Accumulate losses
                for k, v in step_losses.items():
                    running_losses[k] = running_losses.get(k, 0) + v

                self.step += 1

                # Logging
                if self.step % self.config.log_interval == 0:
                    avg_losses = {k: v / self.config.log_interval for k, v in running_losses.items()}
                    elapsed = time.time() - start_time
                    steps_per_sec = self.step / elapsed

                    print(f"Step {self.step}/{self.config.max_steps} | "
                          f"Loss: {avg_losses['total']:.4f} | "
                          f"Ordinal: {avg_losses.get('ordinal', 0):.4f} | "
                          f"Value: {avg_losses.get('value', 0):.4f} | "
                          f"LR: {self.scheduler.get_last_lr()[0]:.2e} | "
                          f"{steps_per_sec:.1f} steps/s")

                    running_losses = {}

                # Evaluation
                if self.step % self.config.eval_interval == 0:
                    eval_metrics = self.evaluate(val_loader)
                    self.metrics_history.append({
                        'step': self.step,
                        **eval_metrics
                    })

                    print(f"\n=== Evaluation at step {self.step} ===")
                    print(f"Ordinal Accuracy: {eval_metrics['ordinal_accuracy']:.4f}")
                    print(f"Kendall's Tau: {eval_metrics['ordinal_kendall_tau']:.4f}")
                    print(f"Value Accuracy: {eval_metrics['value_accuracy']:.4f}")
                    for tier in OrdinalTier:
                        key = f'accuracy_{tier.name.lower()}'
                        if key in eval_metrics:
                            print(f"  {tier.name}: {eval_metrics[key]:.4f}")
                    print()

                # Checkpointing
                if self.step % self.config.checkpoint_interval == 0:
                    self.save_checkpoint()

                if self.step >= self.config.max_steps:
                    break

        # Final checkpoint
        self.save_checkpoint()
        print(f"Training complete! Final step: {self.step}")


def main():
    """Main training entry point."""
    if not HAS_TORCH:
        print("PyTorch required for training. Install with: pip install torch")
        return

    # Configuration
    config = TrainingConfig(
        model_size="small",
        batch_size=32,
        max_steps=10000,
        n_train_samples=50000,
        n_val_samples=5000,
        log_interval=50,
        eval_interval=500,
        checkpoint_interval=2000,
    )

    print("=== OrdinalChess Training ===\n")
    print(f"Configuration:")
    print(f"  Model size: {config.model_size}")
    print(f"  Device: {config.device}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Max steps: {config.max_steps}")
    print()

    # Create dataloaders
    print("Creating datasets...")
    train_loader, val_loader = create_dataloaders(
        n_train=config.n_train_samples,
        n_val=config.n_val_samples,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    # Analyze training data distribution
    print("\nTraining data distribution:")
    train_stats = analyze_dataset(train_loader.dataset)
    for tier, count in train_stats.items():
        pct = 100 * count / len(train_loader.dataset)
        print(f"  {tier}: {count} ({pct:.1f}%)")
    print()

    # Initialize trainer and start training
    trainer = Trainer(config)
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
