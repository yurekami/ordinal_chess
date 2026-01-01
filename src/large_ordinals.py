"""
Large Ordinals and Continuum Cardinals

Extends the ordinal system to handle:
1. Ordinals beyond ω³ (Veblen hierarchy)
2. Ordinal notations for countable ordinals
3. Continuum cardinality (2^ℵ₀) for infinite-dimensional chess
4. Cardinal arithmetic for game-theoretic analysis

Based on:
- Bolan-Tsevas: Every ordinal ≤ continuum arises in infinite-dim chess
- Cantor's theorem: |2^ω| = |ℝ| = c (continuum)
- Veblen functions for ordinal notation

Key Result: With infinite-dimensional chess and full piece rules,
game values can achieve ANY ordinal up to 2^ℵ₀.
With weak pieces (finitely many coord changes), limited to countable ordinals.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union, Callable
from enum import IntEnum, auto
from functools import total_ordering
import math


class CardinalityClass(IntEnum):
    """Classification of ordinal/cardinal sizes."""
    FINITE = 0           # Finite ordinals: 0, 1, 2, ...
    COUNTABLE = 1        # Countable infinite: ℵ₀
    CONTINUUM = 2        # Continuum: 2^ℵ₀ = c = |ℝ|
    BEYOND = 3           # Larger cardinals (theoretical)


@dataclass(frozen=True)
@total_ordering
class LargeOrdinal:
    """
    Representation of large ordinals using Veblen notation.

    The Veblen hierarchy extends Cantor normal form:
    - φ₀(α) = ω^α
    - φ₁(α) = ε_α (epsilon numbers: fixed points of ω^x = x)
    - φ₂(α) = ζ_α (zeta numbers)
    - φ_β(α) for larger β

    Key ordinals:
    - ε₀ = φ₁(0) = sup{ω, ω^ω, ω^ω^ω, ...}
    - Γ₀ = φ(1,0,0) (Feferman-Schütte ordinal)

    For practical purposes, we use a simplified representation.
    """

    # Cantor normal form components (for ordinals < ε₀)
    omega_tower: int = 0      # Height of ω tower (ω^ω^...^ω)
    omega_exponent: int = 0   # Main exponent of ω
    coefficient: int = 1      # Coefficient
    remainder: Optional[LargeOrdinal] = None

    # For larger ordinals
    epsilon_index: int = 0    # ε_n notation
    is_epsilon: bool = False  # True if this is an epsilon number

    # Special markers
    is_zero: bool = False
    is_limit: bool = False    # True if limit ordinal (no predecessor)

    # Cardinality classification
    cardinality: CardinalityClass = CardinalityClass.COUNTABLE

    @classmethod
    def zero(cls) -> LargeOrdinal:
        return cls(is_zero=True, cardinality=CardinalityClass.FINITE)

    @classmethod
    def finite(cls, n: int) -> LargeOrdinal:
        if n == 0:
            return cls.zero()
        return cls(
            omega_exponent=0,
            coefficient=n,
            cardinality=CardinalityClass.FINITE
        )

    @classmethod
    def omega(cls) -> LargeOrdinal:
        """ω = first infinite ordinal."""
        return cls(omega_exponent=1, coefficient=1, is_limit=True)

    @classmethod
    def omega_power(cls, n: int) -> LargeOrdinal:
        """ω^n."""
        return cls(omega_exponent=n, coefficient=1, is_limit=True)

    @classmethod
    def omega_omega(cls) -> LargeOrdinal:
        """ω^ω."""
        return cls(omega_tower=2, is_limit=True)

    @classmethod
    def omega_tower_n(cls, n: int) -> LargeOrdinal:
        """ω^ω^...^ω (tower of height n)."""
        return cls(omega_tower=n, is_limit=True)

    @classmethod
    def epsilon_0(cls) -> LargeOrdinal:
        """ε₀ = sup{ω, ω^ω, ω^ω^ω, ...} = first fixed point of ω^x = x."""
        return cls(is_epsilon=True, epsilon_index=0, is_limit=True)

    @classmethod
    def epsilon_n(cls, n: int) -> LargeOrdinal:
        """ε_n = n-th epsilon number."""
        return cls(is_epsilon=True, epsilon_index=n, is_limit=True)

    @classmethod
    def continuum(cls) -> LargeOrdinal:
        """
        Represents the continuum cardinality c = 2^ℵ₀.

        This is the maximum game value achievable in infinite-dimensional
        chess with unrestricted piece movement.
        """
        return cls(cardinality=CardinalityClass.CONTINUUM, is_limit=True)

    @classmethod
    def aleph_1(cls) -> LargeOrdinal:
        """ω₁ = first uncountable ordinal."""
        return cls(cardinality=CardinalityClass.CONTINUUM, is_limit=True)

    def successor(self) -> LargeOrdinal:
        """Compute α + 1."""
        if self.is_zero:
            return LargeOrdinal.finite(1)

        if self.cardinality == CardinalityClass.FINITE:
            return LargeOrdinal.finite(self.coefficient + 1)

        # For infinite ordinals, successor is same cardinality
        return LargeOrdinal(
            omega_tower=self.omega_tower,
            omega_exponent=self.omega_exponent,
            coefficient=self.coefficient,
            remainder=LargeOrdinal.finite(1) if self.remainder is None else self.remainder.successor(),
            epsilon_index=self.epsilon_index,
            is_epsilon=self.is_epsilon,
            is_zero=False,
            is_limit=False,
            cardinality=self.cardinality,
        )

    def __lt__(self, other: LargeOrdinal) -> bool:
        """Ordinal comparison."""
        # Compare by cardinality first
        if self.cardinality != other.cardinality:
            return self.cardinality < other.cardinality

        # Handle zeros
        if self.is_zero:
            return not other.is_zero
        if other.is_zero:
            return False

        # Handle epsilon numbers
        if self.is_epsilon and other.is_epsilon:
            return self.epsilon_index < other.epsilon_index
        if self.is_epsilon:
            return False  # ε_n > any ω tower
        if other.is_epsilon:
            return True

        # Compare omega towers
        if self.omega_tower != other.omega_tower:
            return self.omega_tower < other.omega_tower

        # Compare exponents
        if self.omega_exponent != other.omega_exponent:
            return self.omega_exponent < other.omega_exponent

        # Compare coefficients
        if self.coefficient != other.coefficient:
            return self.coefficient < other.coefficient

        # Compare remainders
        if self.remainder is None and other.remainder is None:
            return False
        if self.remainder is None:
            return True
        if other.remainder is None:
            return False
        return self.remainder < other.remainder

    def __eq__(self, other) -> bool:
        if not isinstance(other, LargeOrdinal):
            return False
        return (self.omega_tower == other.omega_tower and
                self.omega_exponent == other.omega_exponent and
                self.coefficient == other.coefficient and
                self.remainder == other.remainder and
                self.epsilon_index == other.epsilon_index and
                self.is_epsilon == other.is_epsilon and
                self.is_zero == other.is_zero and
                self.cardinality == other.cardinality)

    def __hash__(self) -> int:
        return hash((self.omega_tower, self.omega_exponent, self.coefficient,
                    self.epsilon_index, self.is_epsilon, self.is_zero,
                    self.cardinality))

    def is_countable(self) -> bool:
        """Check if ordinal is countable (< ω₁)."""
        return self.cardinality <= CardinalityClass.COUNTABLE

    def is_achievable_weak_pieces(self) -> bool:
        """Check if ordinal is achievable with weak pieces rule."""
        # Weak pieces = only finitely many coordinates change per move
        # This limits us to countable ordinals
        return self.is_countable()

    def is_achievable_full_pieces(self) -> bool:
        """Check if ordinal is achievable with full pieces."""
        # Full infinite-dimensional chess achieves up to continuum
        return self.cardinality <= CardinalityClass.CONTINUUM

    def __repr__(self) -> str:
        if self.is_zero:
            return "0"

        if self.cardinality == CardinalityClass.CONTINUUM:
            return "𝔠 (continuum)"

        if self.is_epsilon:
            return f"ε_{self.epsilon_index}"

        if self.omega_tower > 1:
            tower = "^".join(["ω"] * self.omega_tower)
            base = tower
        elif self.omega_exponent > 0:
            if self.omega_exponent == 1:
                base = "ω"
            else:
                base = f"ω^{self.omega_exponent}"
        else:
            base = ""

        if base:
            if self.coefficient > 1:
                result = f"{base}·{self.coefficient}"
            else:
                result = base
        else:
            result = str(self.coefficient)

        if self.remainder and not self.remainder.is_zero:
            result += f" + {self.remainder}"

        return result


class OrdinalArithmetic:
    """Operations on ordinals."""

    @staticmethod
    def add(a: LargeOrdinal, b: LargeOrdinal) -> LargeOrdinal:
        """
        Ordinal addition α + β.

        Note: Ordinal addition is NOT commutative for transfinite ordinals!
        1 + ω = ω, but ω + 1 = ω + 1 ≠ ω
        """
        if a.is_zero:
            return b
        if b.is_zero:
            return a

        # Larger cardinality dominates
        if b.cardinality > a.cardinality:
            return b

        if a.cardinality == CardinalityClass.FINITE:
            if b.cardinality == CardinalityClass.FINITE:
                return LargeOrdinal.finite(a.coefficient + b.coefficient)
            return b  # n + ω = ω

        # Both infinite
        # α + β where β ≥ ω^(exp of α) → β absorbs α
        if b.omega_exponent >= a.omega_exponent:
            return b

        # Otherwise add to remainder
        new_remainder = OrdinalArithmetic.add(
            a.remainder if a.remainder else LargeOrdinal.zero(),
            b
        )
        return LargeOrdinal(
            omega_tower=a.omega_tower,
            omega_exponent=a.omega_exponent,
            coefficient=a.coefficient,
            remainder=new_remainder,
            is_epsilon=a.is_epsilon,
            epsilon_index=a.epsilon_index,
            is_limit=b.is_limit,
            cardinality=max(a.cardinality, b.cardinality),
        )

    @staticmethod
    def multiply(a: LargeOrdinal, b: LargeOrdinal) -> LargeOrdinal:
        """
        Ordinal multiplication α · β.

        ω · n = ω + ω + ... + ω (n times) = ω
        n · ω = ω
        ω · ω = ω²
        """
        if a.is_zero or b.is_zero:
            return LargeOrdinal.zero()

        if a.cardinality == CardinalityClass.FINITE and b.cardinality == CardinalityClass.FINITE:
            return LargeOrdinal.finite(a.coefficient * b.coefficient)

        if a.cardinality == CardinalityClass.FINITE:
            # n · ω^β = ω^β
            return b

        if b.cardinality == CardinalityClass.FINITE:
            # ω^α · n = ω^α + ω^α + ... = ω^α if n ≥ 1
            return a

        # Both transfinite: ω^α · ω^β = ω^(α+β)
        new_exponent = a.omega_exponent + b.omega_exponent
        return LargeOrdinal(
            omega_exponent=new_exponent,
            coefficient=1,
            is_limit=True,
            cardinality=max(a.cardinality, b.cardinality),
        )

    @staticmethod
    def power(base: LargeOrdinal, exp: LargeOrdinal) -> LargeOrdinal:
        """
        Ordinal exponentiation α^β.

        ω^ω = ω^ω
        ω^(ω^ω) = ω tower of 3
        """
        if exp.is_zero:
            return LargeOrdinal.finite(1)

        if base.is_zero:
            return LargeOrdinal.zero()

        if base.cardinality == CardinalityClass.FINITE and exp.cardinality == CardinalityClass.FINITE:
            return LargeOrdinal.finite(base.coefficient ** exp.coefficient)

        # ω^α handling
        if base.omega_exponent == 1 and base.coefficient == 1 and base.omega_tower == 0:
            # base is ω
            if exp.cardinality == CardinalityClass.FINITE:
                return LargeOrdinal.omega_power(exp.coefficient)
            if exp.omega_exponent == 1 and exp.coefficient == 1:
                # ω^ω
                return LargeOrdinal.omega_omega()
            # ω^(ω^n) = tower of n+1
            return LargeOrdinal.omega_tower_n(exp.omega_exponent + 1)

        # General case: use tower notation
        return LargeOrdinal.omega_tower_n(exp.omega_exponent + base.omega_tower)


class GameValueTheory:
    """
    Theoretical framework for game values in infinite-dimensional chess.

    Key Results (Bolan-Tsevas 2024):
    1. With full pieces: achievable values = all ordinals ≤ continuum
    2. With weak pieces: achievable values = all countable ordinals
    3. Board sidelength ≥ 3 is sufficient
    4. Maximum value equals number of possible moves from any position
    """

    @staticmethod
    def max_value_full_pieces() -> LargeOrdinal:
        """Maximum game value with unrestricted pieces."""
        return LargeOrdinal.continuum()

    @staticmethod
    def max_value_weak_pieces() -> LargeOrdinal:
        """Maximum game value with weak pieces constraint."""
        return LargeOrdinal.aleph_1()  # ω₁ (first uncountable)

    @staticmethod
    def moves_per_position_full() -> str:
        """Cardinality of moves from a position with full pieces."""
        return "2^ℵ₀ (continuum)"

    @staticmethod
    def moves_per_position_weak() -> str:
        """Cardinality of moves from a position with weak pieces."""
        return "ℵ₀ (countable)"

    @staticmethod
    def is_achievable(
        ordinal: LargeOrdinal,
        weak_pieces: bool = False
    ) -> bool:
        """Check if ordinal is achievable in infinite-dimensional chess."""
        if weak_pieces:
            return ordinal.is_achievable_weak_pieces()
        return ordinal.is_achievable_full_pieces()

    @staticmethod
    def required_dimensions(ordinal: LargeOrdinal) -> str:
        """Estimate dimensions needed to represent ordinal."""
        if ordinal.cardinality == CardinalityClass.FINITE:
            return f"finite ({ordinal.coefficient} dimensions suffice)"
        if ordinal.cardinality == CardinalityClass.COUNTABLE:
            if ordinal.is_epsilon:
                return "countably infinite (ℵ₀)"
            return f"≈ ω^{ordinal.omega_exponent} dimensions"
        return "continuum many (2^ℵ₀)"


# Common ordinals for reference
ORDINAL_CONSTANTS = {
    "zero": LargeOrdinal.zero(),
    "one": LargeOrdinal.finite(1),
    "omega": LargeOrdinal.omega(),
    "omega_squared": LargeOrdinal.omega_power(2),
    "omega_cubed": LargeOrdinal.omega_power(3),
    "omega_omega": LargeOrdinal.omega_omega(),
    "epsilon_0": LargeOrdinal.epsilon_0(),
    "continuum": LargeOrdinal.continuum(),
}


if __name__ == "__main__":
    print("=== Large Ordinals Test ===\n")

    print("1. Basic Ordinals:")
    for name, ordinal in ORDINAL_CONSTANTS.items():
        print(f"   {name}: {ordinal}")

    print("\n2. Ordinal Comparisons:")
    omega = LargeOrdinal.omega()
    omega2 = LargeOrdinal.omega_power(2)
    omega_omega = LargeOrdinal.omega_omega()
    eps0 = LargeOrdinal.epsilon_0()

    print(f"   ω < ω²: {omega < omega2}")
    print(f"   ω² < ω^ω: {omega2 < omega_omega}")
    print(f"   ω^ω < ε₀: {omega_omega < eps0}")

    print("\n3. Ordinal Arithmetic:")
    result = OrdinalArithmetic.add(omega, LargeOrdinal.finite(5))
    print(f"   ω + 5 = {result}")

    result = OrdinalArithmetic.multiply(omega, omega)
    print(f"   ω · ω = {result}")

    print("\n4. Game Value Theory:")
    print(f"   Max value (full pieces): {GameValueTheory.max_value_full_pieces()}")
    print(f"   Max value (weak pieces): {GameValueTheory.max_value_weak_pieces()}")
    print(f"   Moves per position (full): {GameValueTheory.moves_per_position_full()}")

    print("\n5. Achievability:")
    for name, ordinal in list(ORDINAL_CONSTANTS.items())[:5]:
        achievable = GameValueTheory.is_achievable(ordinal, weak_pieces=False)
        print(f"   {name} achievable (full): {achievable}")

    print("\n=== Tests Complete ===")
