"""
Tests for ordinal value representation and operations.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '..')

from src.ordinals import OrdinalValue, OrdinalTier, OrdinalBucketizer


class TestOrdinalValue:
    """Tests for OrdinalValue class."""

    def test_finite_ordinal(self):
        """Test finite ordinal creation and properties."""
        o = OrdinalValue.from_ply(42)
        assert o.finite == 42
        assert o.omega == 0
        assert o.is_finite()
        assert not o.is_transfinite()
        assert o.tier() == OrdinalTier.FINITE
        assert str(o) == "42"

    def test_omega_ordinal(self):
        """Test omega ordinal creation."""
        o = OrdinalValue.from_omega(1, 0)
        assert o.omega == 1
        assert o.finite == 0
        assert o.is_transfinite()
        assert not o.is_finite()
        assert o.tier() == OrdinalTier.OMEGA
        assert str(o) == "ω"

    def test_omega_plus_n(self):
        """Test omega + n ordinal."""
        o = OrdinalValue.from_omega(1, 5)
        assert o.omega == 1
        assert o.finite == 5
        assert o.tier() == OrdinalTier.OMEGA_PLUS
        assert str(o) == "ω + 5"

    def test_omega_times_n(self):
        """Test omega × n ordinal."""
        o = OrdinalValue.from_omega(3, 0)
        assert o.omega == 3
        assert o.tier() == OrdinalTier.OMEGA_TIMES
        assert str(o) == "ω·3"

    def test_omega_squared(self):
        """Test omega squared ordinal."""
        o = OrdinalValue.from_omega_squared(1, 0, 0)
        assert o.omega_squared == 1
        assert o.tier() == OrdinalTier.OMEGA_SQUARED
        assert str(o) == "ω²"

    def test_omega_cubed(self):
        """Test omega cubed ordinal."""
        o = OrdinalValue.from_omega_cubed(1, 0, 0, 0)
        assert o.omega_cubed == 1
        assert o.tier() == OrdinalTier.OMEGA_CUBED
        assert str(o) == "ω³"

    def test_ordinal_comparison(self):
        """Test ordinal comparison operators."""
        finite = OrdinalValue.from_ply(1000)
        omega = OrdinalValue.from_omega(1, 0)
        omega_sq = OrdinalValue.from_omega_squared(1, 0, 0)

        # Finite < omega < omega²
        assert finite < omega
        assert omega < omega_sq
        assert finite < omega_sq

        # Reflexive
        assert omega <= omega
        assert omega >= omega

        # Same tier comparison
        small = OrdinalValue.from_ply(10)
        large = OrdinalValue.from_ply(100)
        assert small < large

    def test_draw_and_loss(self):
        """Test special values: draw and loss."""
        draw = OrdinalValue.draw()
        loss = OrdinalValue.loss()

        assert draw.is_draw
        assert loss.is_loss
        assert str(draw) == "Draw"
        assert str(loss) == "Loss"

    def test_successor(self):
        """Test successor ordinal."""
        o = OrdinalValue.from_ply(5)
        succ = o.successor()
        assert succ.finite == 6

        omega = OrdinalValue.from_omega(1, 0)
        omega_succ = omega.successor()
        assert omega_succ.omega == 1
        assert omega_succ.finite == 1


class TestOrdinalBucketizer:
    """Tests for OrdinalBucketizer class."""

    def test_bucket_count(self):
        """Test bucket count calculation."""
        b = OrdinalBucketizer()
        # Should have: 201 finite + 10 omega+ + 10 omega× + 5 ω² + 3 ω³ + 2 special
        assert b.n_buckets == 231

    def test_finite_bucketing(self):
        """Test finite ordinal bucketing."""
        b = OrdinalBucketizer()

        for ply in [0, 1, 50, 100, 200]:
            ordinal = OrdinalValue.from_ply(ply)
            bucket = b.to_bucket(ordinal)
            assert bucket == ply

    def test_omega_bucketing(self):
        """Test omega ordinal bucketing."""
        b = OrdinalBucketizer()

        omega = OrdinalValue.from_omega(1, 0)
        bucket = b.to_bucket(omega)
        assert bucket == b.offset_omega_plus

    def test_round_trip(self):
        """Test bucket -> ordinal -> bucket round trip."""
        b = OrdinalBucketizer()

        test_ordinals = [
            OrdinalValue.from_ply(42),
            OrdinalValue.from_omega(1, 3),
            OrdinalValue.from_omega(5, 0),
            OrdinalValue.from_omega_squared(2, 0, 0),
            OrdinalValue.draw(),
            OrdinalValue.loss(),
        ]

        for ordinal in test_ordinals:
            bucket = b.to_bucket(ordinal)
            recovered = b.from_bucket(bucket)
            # Tier should match
            assert recovered.tier() == ordinal.tier()

    def test_cumulative_encoding(self):
        """Test cumulative probability encoding."""
        b = OrdinalBucketizer()

        ordinal = OrdinalValue.from_ply(50)
        cumulative = b.to_cumulative(ordinal)

        # Should have 1s up to bucket 49, 0s after
        assert cumulative.shape[0] == b.n_buckets - 1
        assert cumulative[:50].sum() == 50
        assert cumulative[50:].sum() == 0


class TestOrdinalOrdering:
    """Tests for ordinal ordering properties."""

    def test_natural_number_ordering(self):
        """Test that finite ordinals preserve natural number ordering."""
        ordinals = [OrdinalValue.from_ply(i) for i in range(100)]

        for i in range(99):
            assert ordinals[i] < ordinals[i + 1]

    def test_omega_dominates_finite(self):
        """Test that ω > n for all finite n."""
        omega = OrdinalValue.from_omega(1, 0)

        for n in [0, 1, 10, 100, 1000, 10000]:
            finite = OrdinalValue.from_ply(n)
            assert finite < omega

    def test_omega_squared_dominates_omega(self):
        """Test that ω² > ω·n for all finite n."""
        omega_sq = OrdinalValue.from_omega_squared(1, 0, 0)

        for n in range(1, 20):
            omega_n = OrdinalValue.from_omega(n, 0)
            assert omega_n < omega_sq

    def test_transfinite_ordering(self):
        """Test complete transfinite ordering."""
        ordinals = [
            OrdinalValue.from_ply(100),
            OrdinalValue.from_omega(1, 0),
            OrdinalValue.from_omega(1, 5),
            OrdinalValue.from_omega(2, 0),
            OrdinalValue.from_omega(10, 0),
            OrdinalValue.from_omega_squared(1, 0, 0),
            OrdinalValue.from_omega_squared(1, 5, 0),
            OrdinalValue.from_omega_squared(2, 0, 0),
            OrdinalValue.from_omega_cubed(1, 0, 0, 0),
        ]

        for i in range(len(ordinals) - 1):
            assert ordinals[i] < ordinals[i + 1], \
                f"{ordinals[i]} should be < {ordinals[i+1]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
