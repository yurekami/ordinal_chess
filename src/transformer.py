"""
OrdinalChess Transformer Architecture

A decoder-only transformer with multiple prediction heads:
1. Value Head: Win probability estimation (standard)
2. Action-Value Head: Per-move evaluation (from searchless_chess)
3. Policy Head: Move probability distribution (behavioral cloning)
4. Ordinal Head: Transfinite game value prediction (novel contribution)

The ordinal head implements the H1/H3 hypotheses from our analysis:
- Cumulative ordinal regression for ordered bucket prediction
- Separate finite ply regressor + transfinite classifier (two-head scheme)

Architecture adapted from Google DeepMind's searchless_chess paper
with extensions for ordinal value estimation.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import numpy as np

# Neural network framework imports (supporting both JAX and PyTorch)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    # Stub classes for documentation
    class nn:
        class Module:
            pass

from .ordinals import OrdinalBucketizer, OrdinalValue, OrdinalTier


@dataclass
class TransformerConfig:
    """Configuration for OrdinalChess Transformer."""

    # Model size (following searchless_chess sizing)
    vocab_size: int = 13          # 0=empty, 1-6=white, 7-12=black pieces
    n_embed: int = 256            # Embedding dimension
    n_heads: int = 8              # Attention heads
    n_layers: int = 8             # Transformer layers
    dropout: float = 0.1         # Dropout rate

    # Board representation
    max_board_size: int = 64      # Maximum board dimension (for pos encoding)
    window_size: int = 16         # Observation window size

    # Ordinal prediction settings
    max_finite_ply: int = 200     # Maximum finite ply to predict
    n_ordinal_buckets: int = 232  # Total ordinal buckets (from OrdinalBucketizer)
    use_cumulative_ordinal: bool = True  # Use cumulative link for ordinal regression

    # Action space
    n_moves: int = 4672           # Maximum legal moves (standard chess)

    # Training
    use_flash_attention: bool = True

    @classmethod
    def small(cls) -> TransformerConfig:
        """9M parameter configuration."""
        return cls(n_embed=256, n_heads=4, n_layers=4)

    @classmethod
    def medium(cls) -> TransformerConfig:
        """70M parameter configuration."""
        return cls(n_embed=512, n_heads=8, n_layers=8)

    @classmethod
    def large(cls) -> TransformerConfig:
        """270M parameter configuration (searchless_chess flagship)."""
        return cls(n_embed=1024, n_heads=16, n_layers=16)


class RelativePositionEmbedding(nn.Module):
    """
    Relative position embeddings for extended/infinite boards.

    Instead of absolute position embeddings (which fail for infinite boards),
    we use relative position encodings based on displacement from the
    observation center (typically the player's king).

    This implements hypothesis H4: "Transformer needs sparse/relative
    positional encodings to stably learn on extended/infinite boards."
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        max_rel = config.max_board_size

        # Separate embeddings for file and rank displacements
        self.file_embed = nn.Embedding(2 * max_rel + 1, config.n_embed // 2)
        self.rank_embed = nn.Embedding(2 * max_rel + 1, config.n_embed // 2)

    def forward(self, rel_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rel_positions: (batch, seq_len, 2) with [rel_file, rel_rank]

        Returns:
            Position embeddings of shape (batch, seq_len, n_embed)
        """
        max_rel = self.config.max_board_size

        # Clamp and offset to valid range
        rel_file = rel_positions[..., 0].clamp(-max_rel, max_rel) + max_rel
        rel_rank = rel_positions[..., 1].clamp(-max_rel, max_rel) + max_rel

        file_emb = self.file_embed(rel_file.long())
        rank_emb = self.rank_embed(rel_rank.long())

        return torch.cat([file_emb, rank_emb], dim=-1)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with optional flash attention."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.n_embed % config.n_heads == 0

        self.n_heads = config.n_heads
        self.n_embed = config.n_embed
        self.head_dim = config.n_embed // config.n_heads
        self.use_flash = config.use_flash_attention

        # Key, Query, Value projections
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed, bias=False)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, C = x.shape

        # Compute Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention
        if self.use_flash and hasattr(F, 'scaled_dot_product_attention'):
            # Use Flash Attention if available (PyTorch 2.0+)
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.dropout.p if self.training else 0,
            )
        else:
            # Standard attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            if attention_mask is not None:
                att = att.masked_fill(attention_mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)
            y = att @ v

        # Reshape and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with pre-layer normalization."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class ValueHead(nn.Module):
    """
    Standard value head predicting win probability.

    Outputs: [loss_prob, draw_prob, win_prob]
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln = nn.LayerNorm(config.n_embed)
        self.fc1 = nn.Linear(config.n_embed, config.n_embed // 2)
        self.fc2 = nn.Linear(config.n_embed // 2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Transformer output for [CLS] token, shape (batch, n_embed)

        Returns:
            Win/draw/loss probabilities, shape (batch, 3)
        """
        x = self.ln(x)
        x = F.gelu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)


class ActionValueHead(nn.Module):
    """
    Action-value head predicting Q(s, a) for each legal move.

    Following searchless_chess, this predicts the value of taking
    each action from the current position.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln = nn.LayerNorm(config.n_embed)
        self.fc1 = nn.Linear(config.n_embed, config.n_embed)
        self.fc2 = nn.Linear(config.n_embed, config.n_moves)

    def forward(self, x: torch.Tensor, legal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Transformer output, shape (batch, n_embed)
            legal_mask: Binary mask of legal moves, shape (batch, n_moves)

        Returns:
            Action values, shape (batch, n_moves)
        """
        x = self.ln(x)
        x = F.gelu(self.fc1(x))
        q = self.fc2(x)

        if legal_mask is not None:
            # Mask illegal moves with large negative value
            q = q.masked_fill(~legal_mask, float('-inf'))

        return q


class PolicyHead(nn.Module):
    """
    Policy head for behavioral cloning (move prediction).

    Outputs probability distribution over legal moves.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln = nn.LayerNorm(config.n_embed)
        self.fc1 = nn.Linear(config.n_embed, config.n_embed)
        self.fc2 = nn.Linear(config.n_embed, config.n_moves)

    def forward(self, x: torch.Tensor, legal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Transformer output, shape (batch, n_embed)
            legal_mask: Binary mask of legal moves, shape (batch, n_moves)

        Returns:
            Move probabilities, shape (batch, n_moves)
        """
        x = self.ln(x)
        x = F.gelu(self.fc1(x))
        logits = self.fc2(x)

        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask, float('-inf'))

        return F.softmax(logits, dim=-1)


class OrdinalHead(nn.Module):
    """
    Ordinal value prediction head for transfinite game values.

    Implements two approaches (configurable):

    1. Cumulative Ordinal Regression (H1):
       Predicts P(ordinal > threshold_k) for each threshold k.
       Final prediction is the highest k where P > 0.5.

    2. Two-Head Scheme (H3):
       - Finite ply regressor: Predicts exact ply count when game is finite
       - Transfinite classifier: Classifies among {ω, ω+k, ω·n, ω², ω³}
       - Gate: Predicts whether position has finite or transfinite value
    """

    def __init__(self, config: TransformerConfig, bucketizer: OrdinalBucketizer):
        super().__init__()
        self.config = config
        self.bucketizer = bucketizer
        self.use_cumulative = config.use_cumulative_ordinal

        # Shared preprocessing
        self.ln = nn.LayerNorm(config.n_embed)
        self.fc_shared = nn.Linear(config.n_embed, config.n_embed // 2)

        if self.use_cumulative:
            # Cumulative ordinal regression
            # Predict P(ordinal > k) for each threshold k
            self.fc_cumulative = nn.Linear(
                config.n_embed // 2,
                bucketizer.n_buckets - 1  # K-1 thresholds for K buckets
            )
        else:
            # Two-head scheme
            # Gate: P(finite) vs P(transfinite)
            self.fc_gate = nn.Linear(config.n_embed // 2, 2)

            # Finite ply regressor
            self.fc_ply = nn.Linear(config.n_embed // 2, 1)

            # Transfinite tier classifier
            n_transfinite = (bucketizer.n_omega_plus + bucketizer.n_omega_times +
                           bucketizer.n_omega_sq + bucketizer.n_omega_cube + 2)
            self.fc_transfinite = nn.Linear(config.n_embed // 2, n_transfinite)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Transformer output for [CLS] token, shape (batch, n_embed)

        Returns:
            Dictionary with prediction tensors:
            - 'bucket_probs': Probability distribution over ordinal buckets
            - 'predicted_bucket': Argmax bucket index
            - 'is_transfinite': Probability of transfinite value
            (if not cumulative):
            - 'finite_ply': Predicted ply count (if finite)
            - 'transfinite_probs': Distribution over transfinite tiers
        """
        x = self.ln(x)
        h = F.gelu(self.fc_shared(x))

        results = {}

        if self.use_cumulative:
            # Cumulative logits: P(ordinal > threshold_k)
            cumulative_logits = self.fc_cumulative(h)
            cumulative_probs = torch.sigmoid(cumulative_logits)

            # Convert to bucket probabilities
            # P(bucket k) = P(ordinal > k-1) - P(ordinal > k)
            # With boundary conditions: P(ordinal > -1) = 1, P(ordinal > K-1) = 0
            ones = torch.ones(x.shape[0], 1, device=x.device)
            zeros = torch.zeros(x.shape[0], 1, device=x.device)
            extended_probs = torch.cat([ones, cumulative_probs, zeros], dim=1)
            bucket_probs = extended_probs[:, :-1] - extended_probs[:, 1:]
            bucket_probs = F.relu(bucket_probs)  # Ensure non-negative
            bucket_probs = bucket_probs / (bucket_probs.sum(dim=1, keepdim=True) + 1e-8)

            results['bucket_probs'] = bucket_probs
            results['predicted_bucket'] = bucket_probs.argmax(dim=1)
            results['cumulative_probs'] = cumulative_probs

            # Estimate if transfinite
            transfinite_start = self.bucketizer.offset_omega_plus
            results['is_transfinite'] = bucket_probs[:, transfinite_start:].sum(dim=1)

        else:
            # Two-head scheme
            # Gate prediction
            gate_logits = self.fc_gate(h)
            gate_probs = F.softmax(gate_logits, dim=1)
            results['is_transfinite'] = gate_probs[:, 1]

            # Finite ply prediction (always compute, weight by gate)
            ply_pred = F.relu(self.fc_ply(h).squeeze(-1)) * self.config.max_finite_ply
            results['finite_ply'] = ply_pred

            # Transfinite tier prediction
            transfinite_logits = self.fc_transfinite(h)
            transfinite_probs = F.softmax(transfinite_logits, dim=1)
            results['transfinite_probs'] = transfinite_probs

            # Combine into bucket probabilities
            bucket_probs = self._combine_predictions(
                gate_probs, ply_pred, transfinite_probs
            )
            results['bucket_probs'] = bucket_probs
            results['predicted_bucket'] = bucket_probs.argmax(dim=1)

        return results

    def _combine_predictions(
        self,
        gate_probs: torch.Tensor,
        ply_pred: torch.Tensor,
        transfinite_probs: torch.Tensor
    ) -> torch.Tensor:
        """Combine two-head predictions into bucket distribution."""
        batch_size = gate_probs.shape[0]
        device = gate_probs.device
        bucket_probs = torch.zeros(batch_size, self.bucketizer.n_buckets, device=device)

        # Finite component: Gaussian around predicted ply
        for i in range(self.bucketizer.n_finite):
            # Soft assignment based on distance from prediction
            dist = (i - ply_pred).abs()
            weight = torch.exp(-dist / 10)  # Spread over nearby buckets
            bucket_probs[:, i] = gate_probs[:, 0] * weight

        # Transfinite component
        offset = self.bucketizer.offset_omega_plus
        n_trans = self.bucketizer.n_buckets - offset
        bucket_probs[:, offset:] = gate_probs[:, 1:2] * transfinite_probs[:, :n_trans]

        # Normalize
        bucket_probs = bucket_probs / (bucket_probs.sum(dim=1, keepdim=True) + 1e-8)

        return bucket_probs


class OrdinalChessTransformer(nn.Module):
    """
    Complete OrdinalChess Transformer model.

    Combines:
    - Decoder-only transformer backbone (from searchless_chess)
    - Relative position embeddings (for infinite boards)
    - Multiple prediction heads:
      - Value head (win/draw/loss)
      - Action-value head (Q-values)
      - Policy head (move probabilities)
      - Ordinal head (transfinite game values)
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Initialize bucketizer
        self.bucketizer = OrdinalBucketizer(max_finite_ply=config.max_finite_ply)

        # Token embeddings
        self.tok_embed = nn.Embedding(config.vocab_size, config.n_embed)

        # Relative position embeddings
        self.pos_embed = RelativePositionEmbedding(config)

        # Special tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.n_embed))
        self.turn_embed = nn.Embedding(2, config.n_embed)  # White/Black turn

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.ln_f = nn.LayerNorm(config.n_embed)

        # Prediction heads
        self.value_head = ValueHead(config)
        self.action_value_head = ActionValueHead(config)
        self.policy_head = PolicyHead(config)
        self.ordinal_head = OrdinalHead(config, self.bucketizer)

        # Initialize weights
        self.apply(self._init_weights)

        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"OrdinalChessTransformer initialized with {n_params/1e6:.2f}M parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,           # (batch, window, window) piece tokens
        rel_positions: torch.Tensor,    # (batch, window, window, 2) relative positions
        turn: torch.Tensor,             # (batch,) current turn
        legal_mask: Optional[torch.Tensor] = None,  # (batch, n_moves) legal moves
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Returns dictionary with all prediction heads' outputs.
        """
        B = tokens.shape[0]
        W = self.config.window_size

        # Flatten spatial dimensions
        tokens_flat = tokens.view(B, W * W)  # (batch, seq_len)
        rel_pos_flat = rel_positions.view(B, W * W, 2)

        # Embed tokens and positions
        tok_emb = self.tok_embed(tokens_flat)  # (batch, seq_len, n_embed)
        pos_emb = self.pos_embed(rel_pos_flat)  # (batch, seq_len, n_embed)

        # Combine embeddings
        x = tok_emb + pos_emb

        # Add turn embedding (broadcast to all positions)
        turn_emb = self.turn_embed(turn).unsqueeze(1)  # (batch, 1, n_embed)
        x = x + turn_emb

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)

        x = self.ln_f(x)

        # Extract [CLS] token representation
        cls_repr = x[:, 0]

        # Apply all prediction heads
        results = {
            'value': self.value_head(cls_repr),
            'action_values': self.action_value_head(cls_repr, legal_mask),
            'policy': self.policy_head(cls_repr, legal_mask),
            'hidden': cls_repr,
        }

        # Ordinal head returns a dictionary
        ordinal_results = self.ordinal_head(cls_repr)
        results.update({f'ordinal_{k}': v for k, v in ordinal_results.items()})

        return results

    def predict_ordinal(self, tokens: torch.Tensor, rel_positions: torch.Tensor,
                        turn: torch.Tensor) -> OrdinalValue:
        """
        Predict the ordinal game value for a position.

        Returns:
            OrdinalValue representing the predicted transfinite game value.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(tokens, rel_positions, turn)
            bucket = outputs['ordinal_predicted_bucket'][0].item()
            return self.bucketizer.from_bucket(bucket)


class OrdinalLoss(nn.Module):
    """
    Loss function for ordinal value prediction.

    Combines:
    1. Standard cross-entropy for value/policy heads
    2. Ordinal regression loss with ordering constraints
    3. Consistency regularization between ordinal and value predictions
    """

    def __init__(
        self,
        bucketizer: OrdinalBucketizer,
        value_weight: float = 1.0,
        policy_weight: float = 1.0,
        ordinal_weight: float = 1.0,
        consistency_weight: float = 0.1,
    ):
        super().__init__()
        self.bucketizer = bucketizer
        self.value_weight = value_weight
        self.policy_weight = policy_weight
        self.ordinal_weight = ordinal_weight
        self.consistency_weight = consistency_weight

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute losses for all prediction heads.

        Args:
            predictions: Model outputs from forward()
            targets: Dictionary with target tensors:
                - 'value': (batch,) WDL outcome (0=loss, 1=draw, 2=win)
                - 'policy': (batch,) move index
                - 'ordinal': (batch,) ordinal bucket index

        Returns:
            Dictionary of loss components and total loss.
        """
        losses = {}

        # Value loss (cross-entropy)
        if 'value' in targets:
            value_loss = F.cross_entropy(
                predictions['value'].log(),
                targets['value']
            )
            losses['value'] = self.value_weight * value_loss

        # Policy loss (cross-entropy)
        if 'policy' in targets:
            policy_loss = F.cross_entropy(
                predictions['policy'].log(),
                targets['policy']
            )
            losses['policy'] = self.policy_weight * policy_loss

        # Ordinal loss (cross-entropy on buckets + ordering penalty)
        if 'ordinal' in targets:
            ordinal_probs = predictions['ordinal_bucket_probs']
            ordinal_targets = targets['ordinal']

            # Cross-entropy loss on bucket predictions
            ordinal_ce = F.cross_entropy(
                ordinal_probs.log(),
                ordinal_targets
            )

            # Cumulative ordering loss (if using cumulative regression)
            if 'ordinal_cumulative_probs' in predictions:
                cum_probs = predictions['ordinal_cumulative_probs']
                # Cumulative probabilities should be monotonically decreasing
                # Loss for violations: max(0, P(>k) - P(>k-1))
                ordering_loss = F.relu(cum_probs[:, 1:] - cum_probs[:, :-1]).mean()
                ordinal_loss = ordinal_ce + 0.1 * ordering_loss
            else:
                ordinal_loss = ordinal_ce

            losses['ordinal'] = self.ordinal_weight * ordinal_loss

        # Consistency loss: ordinal predictions should align with value predictions
        if 'value' in targets and 'ordinal' in targets:
            # High ordinal values should correspond to favorable outcomes
            is_transfinite = predictions['ordinal_is_transfinite']
            win_prob = predictions['value'][:, 2]  # P(win)

            # Transfinite positions are typically winning (for the player to move)
            # So high transfinite probability should correlate with high win probability
            consistency_loss = F.mse_loss(is_transfinite, win_prob)
            losses['consistency'] = self.consistency_weight * consistency_loss

        # Total loss
        losses['total'] = sum(losses.values())

        return losses


if __name__ == "__main__":
    if not HAS_TORCH:
        print("PyTorch not available. Install with: pip install torch")
    else:
        print("=== OrdinalChess Transformer Test ===\n")

        # Create small model
        config = TransformerConfig.small()
        model = OrdinalChessTransformer(config)

        # Test forward pass
        batch_size = 2
        tokens = torch.randint(0, config.vocab_size, (batch_size, config.window_size, config.window_size))
        rel_pos = torch.randint(-8, 8, (batch_size, config.window_size, config.window_size, 2))
        turn = torch.randint(0, 2, (batch_size,))

        outputs = model(tokens, rel_pos, turn)

        print("Output shapes:")
        for k, v in outputs.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}")

        # Test ordinal prediction
        ordinal = model.predict_ordinal(tokens[:1], rel_pos[:1], turn[:1])
        print(f"\nPredicted ordinal value: {ordinal}")
        print(f"Ordinal tier: {ordinal.tier().name}")
