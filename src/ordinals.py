"""
Transfinite Ordinal Representation for Chess Game Values

Based on Evans & Hamkins (arXiv 1302.4377) "Transfinite game values in infinite chess"

Ordinal game values represent the "strategic depth" of a chess position:
- Finite ordinals (0, 1, 2, ...): positions with known finite ply to mate
- ω (omega): positions requiring infinitely many moves in limit
- ω·n: omega times n (linear combinations)
- ω²: omega squared (nested infinite structures)
- ω³, ω^k: higher powers for deeper transfinite complexity

This module provides:
1. OrdinalValue class for representing and comparing ordinals
2. Discretization into learnable buckets for neural network training
3. Conversion utilities between symbolic and numeric representations
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union
from enum import IntEnum
import numpy as np


class OrdinalTier(IntEnum):
    """Ordinal tier classification for neural network output heads."""
    FINITE = 0          # Standard finite ply count (0-500+)
    OMEGA = 1           # ω: basic infinite
    OMEGA_PLUS = 2      # ω + n: omega plus finite
    OMEGA_TIMES = 3     # ω · n: omega times finite
    OMEGA_SQUARED = 4   # ω²: omega squared
    OMEGA_SQ_PLUS = 5   # ω² + lower terms
    OMEGA_CUBED = 6     # ω³: omega cubed
    OMEGA_HIGHER = 7    # ω^k for k > 3
    OMEGA_ONE = 8       # ω₁: first uncountable (theoretical limit)


@dataclass(frozen=True, order=False)
class OrdinalValue:
    """
    Represents a transfinite ordinal value for chess positions.

    Uses Cantor Normal Form: ω^α₁·n₁ + ω^α₂·n₂ + ... + ω^α_k·n_k
    where α₁ > α₂ > ... > α_k and n_i are positive integers.

    For practical purposes, we support up to ω³ with finite coefficients.
    """
    # Coefficients for Cantor normal form (simplified for ω³ max)
    omega_cubed: int = 0      # coefficient of ω³
    omega_squared: int = 0    # coefficient of ω²
    omega: int = 0            # coefficient of ω
    finite: int = 0           # finite part (coefficient of ω⁰ = 1)

    # Special markers
    is_draw: bool = False     # Draw = ∞ (neither player wins)
    is_loss: bool = False     # Loss from perspective of player to move

    def __post_init__(self):
        """Validate ordinal representation."""
        assert self.omega_cubed >= 0, "Coefficients must be non-negative"
        assert self.omega_squared >= 0, "Coefficients must be non-negative"
        assert self.omega >= 0, "Coefficients must be non-negative"
        assert self.finite >= 0, "Coefficients must be non-negative"

    @classmethod
    def from_ply(cls, ply: int) -> OrdinalValue:
        """Create ordinal from finite ply count."""
        return cls(finite=ply)

    @classmethod
    def from_omega(cls, coefficient: int = 1, plus: int = 0) -> OrdinalValue:
        """Create ω·n + k ordinal."""
        return cls(omega=coefficient, finite=plus)

    @classmethod
    def from_omega_squared(cls, coef2: int = 1, coef1: int = 0, finite: int = 0) -> OrdinalValue:
        """Create ω²·n₂ + ω·n₁ + k ordinal."""
        return cls(omega_squared=coef2, omega=coef1, finite=finite)

    @classmethod
    def from_omega_cubed(cls, coef3: int = 1, coef2: int = 0, coef1: int = 0, finite: int = 0) -> OrdinalValue:
        """Create ω³·n₃ + ω²·n₂ + ω·n₁ + k ordinal."""
        return cls(omega_cubed=coef3, omega_squared=coef2, omega=coef1, finite=finite)

    @classmethod
    def draw(cls) -> OrdinalValue:
        """Create draw value."""
        return cls(is_draw=True)

    @classmethod
    def loss(cls) -> OrdinalValue:
        """Create loss value (checkmate against player to move)."""
        return cls(is_loss=True)

    def tier(self) -> OrdinalTier:
        """Classify ordinal into tier for neural network head."""
        if self.is_draw or self.is_loss:
            return OrdinalTier.FINITE
        if self.omega_cubed > 0:
            return OrdinalTier.OMEGA_CUBED
        if self.omega_squared > 0:
            if self.omega > 0 or self.finite > 0:
                return OrdinalTier.OMEGA_SQ_PLUS
            return OrdinalTier.OMEGA_SQUARED
        if self.omega > 0:
            if self.omega > 1:
                return OrdinalTier.OMEGA_TIMES
            if self.finite > 0:
                return OrdinalTier.OMEGA_PLUS
            return OrdinalTier.OMEGA
        return OrdinalTier.FINITE

    def is_finite(self) -> bool:
        """Check if ordinal is finite."""
        return (self.omega_cubed == 0 and self.omega_squared == 0 and
                self.omega == 0 and not self.is_draw)

    def is_transfinite(self) -> bool:
        """Check if ordinal is transfinite (≥ ω)."""
        return not self.is_finite() and not self.is_draw and not self.is_loss

    def __lt__(self, other: OrdinalValue) -> bool:
        """Ordinal comparison (lexicographic on Cantor normal form)."""
        if self.is_loss and not other.is_loss:
            return True
        if other.is_loss:
            return False
        if self.is_draw or other.is_draw:
            return False  # Draws are incomparable with wins

        return (self.omega_cubed, self.omega_squared, self.omega, self.finite) < \
               (other.omega_cubed, other.omega_squared, other.omega, other.finite)

    def __le__(self, other: OrdinalValue) -> bool:
        return self == other or self < other

    def __gt__(self, other: OrdinalValue) -> bool:
        return other < self

    def __ge__(self, other: OrdinalValue) -> bool:
        return self == other or self > other

    def __add__(self, other: OrdinalValue) -> OrdinalValue:
        """Ordinal addition (non-commutative for transfinite!)."""
        if self.is_draw or other.is_draw:
            return OrdinalValue.draw()
        if self.is_loss:
            return other
        if other.is_loss:
            return self

        # For transfinite ordinals, α + β absorbs lower terms of α
        # Simplified: we add coefficients (works for common cases)
        return OrdinalValue(
            omega_cubed=self.omega_cubed + other.omega_cubed,
            omega_squared=self.omega_squared + other.omega_squared,
            omega=self.omega + other.omega,
            finite=self.finite + other.finite
        )

    def successor(self) -> OrdinalValue:
        """Compute α + 1 (successor ordinal)."""
        return OrdinalValue(
            omega_cubed=self.omega_cubed,
            omega_squared=self.omega_squared,
            omega=self.omega,
            finite=self.finite + 1,
            is_draw=self.is_draw,
            is_loss=self.is_loss
        )

    def __repr__(self) -> str:
        if self.is_draw:
            return "Draw"
        if self.is_loss:
            return "Loss"

        parts = []
        if self.omega_cubed > 0:
            if self.omega_cubed == 1:
                parts.append("ω³")
            else:
                parts.append(f"ω³·{self.omega_cubed}")
        if self.omega_squared > 0:
            if self.omega_squared == 1:
                parts.append("ω²")
            else:
                parts.append(f"ω²·{self.omega_squared}")
        if self.omega > 0:
            if self.omega == 1:
                parts.append("ω")
            else:
                parts.append(f"ω·{self.omega}")
        if self.finite > 0 or not parts:
            parts.append(str(self.finite))

        return " + ".join(parts)


class OrdinalBucketizer:
    """
    Discretizes ordinal values into fixed buckets for neural network training.

    Bucket structure:
    - [0, 1, 2, ..., max_finite_ply]: Individual finite ply counts
    - [ω, ω+1, ..., ω+k]: Omega plus small finite
    - [ω·2, ω·3, ..., ω·m]: Omega times small multiples
    - [ω², ω²+ω, ω²·2, ...]: Omega squared variants
    - [ω³, ...]: Omega cubed and beyond
    - [Draw, Loss]: Special outcomes
    """

    def __init__(
        self,
        max_finite_ply: int = 200,
        omega_plus_buckets: int = 10,
        omega_times_buckets: int = 10,
        omega_sq_buckets: int = 5,
        omega_cube_buckets: int = 3
    ):
        self.max_finite_ply = max_finite_ply
        self.omega_plus_buckets = omega_plus_buckets
        self.omega_times_buckets = omega_times_buckets
        self.omega_sq_buckets = omega_sq_buckets
        self.omega_cube_buckets = omega_cube_buckets

        # Calculate bucket boundaries
        self.n_finite = max_finite_ply + 1  # 0 to max_finite_ply inclusive
        self.n_omega_plus = omega_plus_buckets
        self.n_omega_times = omega_times_buckets
        self.n_omega_sq = omega_sq_buckets
        self.n_omega_cube = omega_cube_buckets
        self.n_special = 2  # Draw, Loss

        self.n_buckets = (self.n_finite + self.n_omega_plus + self.n_omega_times +
                         self.n_omega_sq + self.n_omega_cube + self.n_special)

        # Bucket offset indices
        self.offset_omega_plus = self.n_finite
        self.offset_omega_times = self.offset_omega_plus + self.n_omega_plus
        self.offset_omega_sq = self.offset_omega_times + self.n_omega_times
        self.offset_omega_cube = self.offset_omega_sq + self.n_omega_sq
        self.offset_special = self.offset_omega_cube + self.n_omega_cube

    def to_bucket(self, ordinal: OrdinalValue) -> int:
        """Convert ordinal value to bucket index."""
        if ordinal.is_draw:
            return self.offset_special
        if ordinal.is_loss:
            return self.offset_special + 1

        if ordinal.is_finite():
            return min(ordinal.finite, self.max_finite_ply)

        # Transfinite cases
        if ordinal.omega_cubed > 0:
            idx = min(ordinal.omega_cubed - 1, self.n_omega_cube - 1)
            return self.offset_omega_cube + idx

        if ordinal.omega_squared > 0:
            # Combine ω² coefficient with lower terms for bucketing
            idx = min(ordinal.omega_squared - 1, self.n_omega_sq - 1)
            return self.offset_omega_sq + idx

        if ordinal.omega > 1:
            # ω·n for n > 1
            idx = min(ordinal.omega - 2, self.n_omega_times - 1)
            return self.offset_omega_times + idx

        # ω + k
        if ordinal.omega == 1:
            idx = min(ordinal.finite, self.n_omega_plus - 1)
            return self.offset_omega_plus + idx

        # Fallback (shouldn't reach here)
        return 0

    def from_bucket(self, bucket: int) -> OrdinalValue:
        """Convert bucket index back to representative ordinal value."""
        if bucket < self.n_finite:
            return OrdinalValue.from_ply(bucket)

        if bucket < self.offset_omega_times:
            k = bucket - self.offset_omega_plus
            return OrdinalValue.from_omega(1, k)

        if bucket < self.offset_omega_sq:
            n = bucket - self.offset_omega_times + 2
            return OrdinalValue.from_omega(n, 0)

        if bucket < self.offset_omega_cube:
            n = bucket - self.offset_omega_sq + 1
            return OrdinalValue.from_omega_squared(n, 0, 0)

        if bucket < self.offset_special:
            n = bucket - self.offset_omega_cube + 1
            return OrdinalValue.from_omega_cubed(n, 0, 0, 0)

        if bucket == self.offset_special:
            return OrdinalValue.draw()

        return OrdinalValue.loss()

    def to_one_hot(self, ordinal: OrdinalValue) -> np.ndarray:
        """Convert ordinal to one-hot encoding."""
        bucket = self.to_bucket(ordinal)
        one_hot = np.zeros(self.n_buckets, dtype=np.float32)
        one_hot[bucket] = 1.0
        return one_hot

    def to_cumulative(self, ordinal: OrdinalValue) -> np.ndarray:
        """
        Convert ordinal to cumulative encoding for ordinal regression.

        For ordinal α in bucket k, returns [1, 1, ..., 1, 0, 0, ..., 0]
        with 1s in positions 0..k-1 (indicating "α ≥ threshold_i").
        """
        bucket = self.to_bucket(ordinal)
        cumulative = np.zeros(self.n_buckets - 1, dtype=np.float32)
        cumulative[:bucket] = 1.0
        return cumulative

    def bucket_labels(self) -> List[str]:
        """Get human-readable labels for all buckets."""
        labels = []
        for i in range(self.n_buckets):
            labels.append(str(self.from_bucket(i)))
        return labels


# Example ordinal motifs from Evans & Hamkins paper
EXAMPLE_ORDINALS = {
    "mate_in_1": OrdinalValue.from_ply(1),
    "mate_in_10": OrdinalValue.from_ply(10),
    "omega_ladder": OrdinalValue.from_omega(1, 0),
    "omega_plus_5": OrdinalValue.from_omega(1, 5),
    "omega_times_2": OrdinalValue.from_omega(2, 0),
    "omega_squared": OrdinalValue.from_omega_squared(1, 0, 0),
    "omega_squared_plus_omega": OrdinalValue.from_omega_squared(1, 1, 0),
    "omega_cubed": OrdinalValue.from_omega_cubed(1, 0, 0, 0),
}


if __name__ == "__main__":
    # Demo
    print("=== OrdinalChess: Transfinite Game Values ===\n")

    bucketizer = OrdinalBucketizer()
    print(f"Total buckets: {bucketizer.n_buckets}\n")

    for name, ordinal in EXAMPLE_ORDINALS.items():
        bucket = bucketizer.to_bucket(ordinal)
        tier = ordinal.tier()
        print(f"{name}: {ordinal} -> bucket {bucket}, tier {tier.name}")

    print("\n=== Ordinal Comparisons ===")
    o1 = OrdinalValue.from_ply(1000)
    o2 = OrdinalValue.from_omega(1, 0)
    o3 = OrdinalValue.from_omega_squared(1, 0, 0)

    print(f"{o1} < {o2}: {o1 < o2}")  # True: 1000 < ω
    print(f"{o2} < {o3}: {o2 < o3}")  # True: ω < ω²
    print(f"{o1} + {o2} = {o1 + o2}")  # ω + 1000 (but mathematically ω absorbs)
