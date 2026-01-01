"""
OrdinalChess: Transfinite Game Value Prediction for Chess

A synthesis of:
1. Google DeepMind's searchless_chess transformer architecture
2. Evans & Hamkins' transfinite ordinal game theory (arXiv 1302.4377)

This package provides:
- OrdinalValue: Representation of transfinite ordinals (ω, ω², ω³, ...)
- SparseBoard: Extended/infinite chess board representation
- OrdinalChessTransformer: Neural network with ordinal prediction heads
- OrdinalChessEngine: Chess engine with ordinal evaluations
- Training and evaluation pipelines

Example usage:
    from ordinal_chess import OrdinalChessEngine, OrdinalValue

    # Create engine
    engine = OrdinalChessEngine()
    engine.new_game()

    # Get evaluation with ordinal value
    evaluation = engine.evaluate()
    print(f"Ordinal value: {evaluation.ordinal_value}")
    print(f"Best move: {evaluation.best_move}")
"""

__version__ = "0.1.0"
__author__ = "OrdinalChess Team"

from .ordinals import (
    OrdinalValue,
    OrdinalTier,
    OrdinalBucketizer,
    EXAMPLE_ORDINALS,
)

from .board import (
    SparseBoard,
    BoardEncoder,
    Move,
    Piece,
    PieceType,
    Color,
)

from .transformer import (
    OrdinalChessTransformer,
    TransformerConfig,
    OrdinalLoss,
)

from .engine import (
    OrdinalChessEngine,
    EngineConfig,
    EngineEvaluation,
)

from .data import (
    OrdinalMotifGenerator,
    OrdinalChessDataset,
    PositionSample,
)

__all__ = [
    # Ordinals
    "OrdinalValue",
    "OrdinalTier",
    "OrdinalBucketizer",
    "EXAMPLE_ORDINALS",
    # Board
    "SparseBoard",
    "BoardEncoder",
    "Move",
    "Piece",
    "PieceType",
    "Color",
    # Model
    "OrdinalChessTransformer",
    "TransformerConfig",
    "OrdinalLoss",
    # Engine
    "OrdinalChessEngine",
    "EngineConfig",
    "EngineEvaluation",
    # Data
    "OrdinalMotifGenerator",
    "OrdinalChessDataset",
    "PositionSample",
]
