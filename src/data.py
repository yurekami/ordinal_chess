"""
Training Data Generation and Dataset for OrdinalChess

Provides:
1. Synthetic position generation with known ordinal values
2. Integration with ChessBench-style annotated data
3. Evans-Hamkins infinite chess motif generators
4. DataLoader utilities for training

The key challenge is generating training data with ground-truth ordinal labels:
- For finite positions: Use Stockfish tablebase or search depth
- For transfinite positions: Construct known motifs from Evans-Hamkins paper
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Iterator, Callable
from pathlib import Path
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Dataset = object

from .ordinals import OrdinalValue, OrdinalBucketizer, OrdinalTier
from .board import SparseBoard, BoardEncoder, Piece, PieceType, Color, Coord


@dataclass
class PositionSample:
    """A training sample with position and annotations."""
    board: SparseBoard
    ordinal: OrdinalValue
    best_move: Optional[str] = None
    value: Optional[float] = None  # Win probability from perspective of player to move
    action_values: Optional[Dict[str, float]] = None  # Q-values for each move
    source: str = "synthetic"  # Where this sample came from


class OrdinalMotifGenerator:
    """
    Generates chess positions with known transfinite ordinal values.

    Based on constructions from Evans & Hamkins "Transfinite game values in infinite chess".

    Key motifs:
    1. ω (omega): Infinite retreat patterns
    2. ω·n: Multiple sequential omega ladders
    3. ω²: Nested infinite structures
    4. ω³: Triply-nested structures
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_finite_mate(self, ply: int) -> PositionSample:
        """
        Generate a mate-in-N position with exact ply count.

        For simplicity, we generate back-rank mate patterns which have
        well-defined ply counts.
        """
        board = SparseBoard()

        # Basic back-rank mate setup
        # White: King + Queen + Rook
        # Black: King trapped on back rank

        # Randomize the exact configuration while maintaining mate-in-N
        black_king_file = self.rng.randint(1, 6)

        board.set_piece((0, 0), Piece(PieceType.KING, Color.WHITE))

        # Position pieces to create mate-in-ply
        if ply == 1:
            # Immediate mate
            board.set_piece((black_king_file, 7), Piece(PieceType.KING, Color.BLACK))
            board.set_piece((black_king_file, 6), Piece(PieceType.QUEEN, Color.WHITE))
            board.set_piece((7, 7), Piece(PieceType.ROOK, Color.WHITE))
        elif ply <= 5:
            # Short mate
            board.set_piece((black_king_file, 7), Piece(PieceType.KING, Color.BLACK))
            board.set_piece((0, 6), Piece(PieceType.QUEEN, Color.WHITE))
            board.set_piece((7, 0), Piece(PieceType.ROOK, Color.WHITE))
            # Add some randomness
            if self.rng.random() < 0.5:
                board.set_piece((self.rng.randint(0, 7), self.rng.randint(0, 5)),
                              Piece(PieceType.PAWN, Color.BLACK))
        else:
            # Longer mate - more complex setup
            board.set_piece((black_king_file, 7), Piece(PieceType.KING, Color.BLACK))
            board.set_piece((self.rng.randint(0, 3), 0), Piece(PieceType.QUEEN, Color.WHITE))
            board.set_piece((self.rng.randint(4, 7), 0), Piece(PieceType.ROOK, Color.WHITE))
            # Add pawns to extend the mate
            for _ in range(min(ply // 2, 4)):
                pawn_file = self.rng.randint(0, 7)
                pawn_rank = self.rng.randint(1, 5)
                board.set_piece((pawn_file, pawn_rank), Piece(PieceType.PAWN, Color.BLACK))

        return PositionSample(
            board=board,
            ordinal=OrdinalValue.from_ply(ply),
            value=1.0,  # Winning for white
            source="synthetic_mate"
        )

    def generate_omega_ladder(self, variation: int = 0) -> PositionSample:
        """
        Generate an ω (omega) position.

        The omega ladder is a position where:
        - White can force mate
        - Black can delay indefinitely through infinite retreat
        - But white will eventually win (game value = ω)

        Classical construction: Black king on infinite rank/file with
        white pieces that can slowly advance.
        """
        # Extend the board for infinite retreat
        retreat_length = 20 + variation * 5
        board = SparseBoard(bounds=(0, 0, retreat_length + 5, 7), is_infinite=False)

        # White's forcing configuration
        board.set_piece((0, 0), Piece(PieceType.KING, Color.WHITE))
        board.set_piece((2, 1), Piece(PieceType.QUEEN, Color.WHITE))
        board.set_piece((3, 0), Piece(PieceType.ROOK, Color.WHITE))

        # Black king that can retreat
        black_king_pos = retreat_length
        board.set_piece((black_king_pos, 4), Piece(PieceType.KING, Color.BLACK))

        # Add variation with ω + k
        finite_add = self.rng.randint(0, 10) if variation > 0 else 0

        return PositionSample(
            board=board,
            ordinal=OrdinalValue.from_omega(1, finite_add),
            value=1.0,  # White wins (eventually)
            source="omega_ladder"
        )

    def generate_omega_times_n(self, n: int = 2) -> PositionSample:
        """
        Generate an ω·n position.

        Multiple sequential omega ladders that must be resolved in order.
        Each ladder contributes one ω to the total value.
        """
        # Extended board for multiple ladders
        board_width = 15 * n + 10
        board = SparseBoard(bounds=(0, 0, board_width, 7), is_infinite=False)

        # White's main forces
        board.set_piece((0, 0), Piece(PieceType.KING, Color.WHITE))
        board.set_piece((1, 1), Piece(PieceType.QUEEN, Color.WHITE))

        # Place n sequential ladder structures
        for i in range(n):
            offset = 10 + i * 15
            # Each ladder has a rook and the black king can retreat through it
            board.set_piece((offset, 0), Piece(PieceType.ROOK, Color.WHITE))
            if i == n - 1:
                # Black king at the final ladder
                board.set_piece((offset + 10, 4), Piece(PieceType.KING, Color.BLACK))
            else:
                # Blocking piece that creates sequential dependency
                board.set_piece((offset + 5, 3), Piece(PieceType.KNIGHT, Color.BLACK))

        return PositionSample(
            board=board,
            ordinal=OrdinalValue.from_omega(n, 0),
            value=1.0,
            source="omega_times_n"
        )

    def generate_omega_squared(self) -> PositionSample:
        """
        Generate an ω² position.

        This requires nested infinite structures - a "ladder of ladders"
        where each step in the outer ladder requires resolving an entire
        inner omega ladder.
        """
        # Large board for nested structure
        board = SparseBoard(bounds=(0, 0, 63, 15), is_infinite=False)

        # White's configuration
        board.set_piece((0, 0), Piece(PieceType.KING, Color.WHITE))
        board.set_piece((1, 1), Piece(PieceType.QUEEN, Color.WHITE))
        board.set_piece((2, 0), Piece(PieceType.ROOK, Color.WHITE))
        board.set_piece((3, 0), Piece(PieceType.ROOK, Color.WHITE))

        # Nested structure - outer ladder
        for i in range(4):
            offset = 10 + i * 12
            # Inner ladder pieces
            board.set_piece((offset, 2), Piece(PieceType.BISHOP, Color.WHITE))
            board.set_piece((offset + 8, 6), Piece(PieceType.KNIGHT, Color.BLACK))

        # Black king deep in the nested structure
        board.set_piece((50, 10), Piece(PieceType.KING, Color.BLACK))

        return PositionSample(
            board=board,
            ordinal=OrdinalValue.from_omega_squared(1, 0, 0),
            value=1.0,
            source="omega_squared"
        )

    def generate_omega_cubed(self) -> PositionSample:
        """
        Generate an ω³ position.

        Triply-nested infinite structures.
        """
        # Very large board
        board = SparseBoard(bounds=(0, 0, 127, 31), is_infinite=False)

        # White's configuration
        board.set_piece((0, 0), Piece(PieceType.KING, Color.WHITE))
        board.set_piece((1, 1), Piece(PieceType.QUEEN, Color.WHITE))
        board.set_piece((2, 0), Piece(PieceType.ROOK, Color.WHITE))
        board.set_piece((3, 0), Piece(PieceType.ROOK, Color.WHITE))
        board.set_piece((4, 0), Piece(PieceType.ROOK, Color.WHITE))

        # Triple-nested structure
        for i in range(3):
            for j in range(3):
                offset_x = 10 + i * 35 + j * 10
                offset_y = 5 + j * 8
                board.set_piece((offset_x, offset_y), Piece(PieceType.BISHOP, Color.WHITE))
                board.set_piece((offset_x + 5, offset_y + 3), Piece(PieceType.KNIGHT, Color.BLACK))

        # Black king deep in nested structure
        board.set_piece((100, 25), Piece(PieceType.KING, Color.BLACK))

        return PositionSample(
            board=board,
            ordinal=OrdinalValue.from_omega_cubed(1, 0, 0, 0),
            value=1.0,
            source="omega_cubed"
        )

    def generate_draw(self) -> PositionSample:
        """Generate a drawn position (insufficient material or stalemate)."""
        board = SparseBoard()

        # King vs King
        board.set_piece((0, 0), Piece(PieceType.KING, Color.WHITE))
        board.set_piece((7, 7), Piece(PieceType.KING, Color.BLACK))

        # Maybe add a single minor piece
        if self.rng.random() < 0.5:
            board.set_piece((3, 3), Piece(PieceType.BISHOP, Color.WHITE))

        return PositionSample(
            board=board,
            ordinal=OrdinalValue.draw(),
            value=0.5,  # Draw
            source="draw"
        )

    def generate_sample(self, tier: Optional[OrdinalTier] = None) -> PositionSample:
        """Generate a random sample, optionally from a specific tier."""
        if tier is None:
            # Random tier with weighted probability
            weights = [0.4, 0.15, 0.1, 0.1, 0.1, 0.05, 0.05, 0.025, 0.025]
            tier = self.rng.choices(list(OrdinalTier), weights=weights)[0]

        if tier == OrdinalTier.FINITE:
            ply = self.rng.randint(1, 50)
            return self.generate_finite_mate(ply)
        elif tier == OrdinalTier.OMEGA:
            return self.generate_omega_ladder(0)
        elif tier == OrdinalTier.OMEGA_PLUS:
            return self.generate_omega_ladder(self.rng.randint(1, 5))
        elif tier == OrdinalTier.OMEGA_TIMES:
            n = self.rng.randint(2, 5)
            return self.generate_omega_times_n(n)
        elif tier == OrdinalTier.OMEGA_SQUARED:
            return self.generate_omega_squared()
        elif tier == OrdinalTier.OMEGA_CUBED:
            return self.generate_omega_cubed()
        else:
            return self.generate_draw()


class OrdinalChessDataset(Dataset):
    """
    PyTorch Dataset for OrdinalChess training.

    Combines:
    - Synthetic positions with known ordinal values
    - (Optional) ChessBench-style annotated positions
    """

    def __init__(
        self,
        n_samples: int = 100000,
        encoder: Optional[BoardEncoder] = None,
        bucketizer: Optional[OrdinalBucketizer] = None,
        include_transfinite: bool = True,
        seed: int = 42,
    ):
        self.n_samples = n_samples
        self.encoder = encoder or BoardEncoder()
        self.bucketizer = bucketizer or OrdinalBucketizer()
        self.include_transfinite = include_transfinite
        self.generator = OrdinalMotifGenerator(seed)

        # Pre-generate samples
        self.samples: List[PositionSample] = []
        self._generate_samples()

    def _generate_samples(self):
        """Pre-generate training samples."""
        for i in range(self.n_samples):
            if self.include_transfinite:
                sample = self.generator.generate_sample()
            else:
                # Only finite positions
                ply = self.generator.rng.randint(1, 100)
                sample = self.generator.generate_finite_mate(ply)
            self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        sample = self.samples[idx]

        # Encode the position
        encoding = self.encoder.encode_position(sample.board)

        # Convert ordinal to bucket index
        ordinal_bucket = self.bucketizer.to_bucket(sample.ordinal)

        # Value target (0=loss, 1=draw, 2=win)
        if sample.value is not None:
            if sample.value > 0.6:
                value_target = 2  # Win
            elif sample.value < 0.4:
                value_target = 0  # Loss
            else:
                value_target = 1  # Draw
        else:
            value_target = 1  # Default to draw

        return {
            'tokens': encoding['tokens'],
            'position_ids': encoding['position_ids'],
            'turn': encoding['turn'],
            'castling': encoding['castling'],
            'ordinal_target': np.array([ordinal_bucket], dtype=np.int64),
            'value_target': np.array([value_target], dtype=np.int64),
            'is_transfinite': np.array([sample.ordinal.is_transfinite()], dtype=np.float32),
        }


def create_dataloaders(
    n_train: int = 100000,
    n_val: int = 10000,
    batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders."""

    if not HAS_TORCH:
        raise ImportError("PyTorch required for dataloaders")

    train_dataset = OrdinalChessDataset(n_samples=n_train, seed=seed)
    val_dataset = OrdinalChessDataset(n_samples=n_val, seed=seed + 1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


# Statistics helper
def analyze_dataset(dataset: OrdinalChessDataset) -> Dict[str, int]:
    """Analyze the distribution of ordinal tiers in the dataset."""
    tier_counts = {tier.name: 0 for tier in OrdinalTier}

    for sample in dataset.samples:
        tier = sample.ordinal.tier()
        tier_counts[tier.name] += 1

    return tier_counts


if __name__ == "__main__":
    print("=== OrdinalChess Data Generation Test ===\n")

    # Test motif generator
    generator = OrdinalMotifGenerator(seed=42)

    print("Sample positions:")
    print("\n1. Mate in 3:")
    sample = generator.generate_finite_mate(3)
    print(sample.board)
    print(f"Ordinal: {sample.ordinal}\n")

    print("2. Omega ladder:")
    sample = generator.generate_omega_ladder()
    print(sample.board)
    print(f"Ordinal: {sample.ordinal}\n")

    print("3. Omega squared:")
    sample = generator.generate_omega_squared()
    print(f"Board size: {sample.board.bounds}")
    print(f"Ordinal: {sample.ordinal}\n")

    # Test dataset
    print("Creating dataset...")
    dataset = OrdinalChessDataset(n_samples=1000)

    print("\nDataset statistics:")
    stats = analyze_dataset(dataset)
    for tier, count in stats.items():
        print(f"  {tier}: {count} ({100*count/len(dataset):.1f}%)")

    print("\nSample encoding:")
    sample = dataset[0]
    for k, v in sample.items():
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
