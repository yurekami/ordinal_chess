"""
OrdinalChess: Transfinite Game Value Prediction for Chess

A synthesis of:
1. Google DeepMind's searchless_chess transformer architecture
2. Evans & Hamkins' transfinite ordinal game theory (arXiv 1302.4377)
3. Bolan & Tsevas' infinite-dimensional chess (2024)

This package provides:
- OrdinalValue: Representation of transfinite ordinals (ω, ω², ω³, ...)
- LargeOrdinal: Extended ordinals up to continuum cardinality
- SparseBoard: Extended/infinite chess board representation
- InfiniteDimensionalBoard: N-dimensional and infinite-dimensional boards
- OrdinalChessTransformer: Neural network with ordinal prediction heads
- TreeEmbedder: Embed ordinal trees into chess positions
- OrdinalChessEngine: Chess engine with ordinal evaluations
- Training and evaluation pipelines

Key Results:
- Standard chess: ordinals up to ω³
- 3D infinite chess: all countable ordinals (Evans-Hamkins)
- Infinite-dimensional chess: all ordinals up to continuum 2^ℵ₀ (Bolan-Tsevas)
- Weak pieces variant: countable ordinals only

Example usage:
    from ordinal_chess import OrdinalChessEngine, OrdinalValue
    from ordinal_chess import InfiniteDimensionalBoard, Coordinate

    # Standard engine
    engine = OrdinalChessEngine()
    engine.new_game()
    evaluation = engine.evaluate()
    print(f"Ordinal value: {evaluation.ordinal_value}")

    # Infinite-dimensional board
    board = InfiniteDimensionalBoard()
    coord = Coordinate.from_list([1, 2, 0, 0, 3])
    print(f"Supremum norm: {coord.sup_norm}")
"""

__version__ = "0.2.0"
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

from .dimensions import (
    Coordinate,
    InfiniteDimensionalBoard,
    BoardConfig,
    BoardVariant,
    InfDimMove,
)

from .tree_embedding import (
    TreeNode,
    TreeBuilder,
    TreeEmbedder,
    OrdinalPositionGenerator,
)

from .large_ordinals import (
    LargeOrdinal,
    CardinalityClass,
    OrdinalArithmetic,
    GameValueTheory,
    ORDINAL_CONSTANTS,
)

__all__ = [
    # Ordinals
    "OrdinalValue",
    "OrdinalTier",
    "OrdinalBucketizer",
    "EXAMPLE_ORDINALS",
    # Large Ordinals
    "LargeOrdinal",
    "CardinalityClass",
    "OrdinalArithmetic",
    "GameValueTheory",
    "ORDINAL_CONSTANTS",
    # Board (2D/3D)
    "SparseBoard",
    "BoardEncoder",
    "Move",
    "Piece",
    "PieceType",
    "Color",
    # Infinite Dimensions
    "Coordinate",
    "InfiniteDimensionalBoard",
    "BoardConfig",
    "BoardVariant",
    "InfDimMove",
    # Tree Embedding
    "TreeNode",
    "TreeBuilder",
    "TreeEmbedder",
    "OrdinalPositionGenerator",
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
