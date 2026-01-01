#!/usr/bin/env python3
"""
OrdinalChess Complete Demo

Demonstrates all features of the OrdinalChess system:
1. Transfinite ordinal values (ω, ω², ω³, ...)
2. Standard 8x8 chess with ordinal evaluation
3. Extended/3D boards (Evans-Hamkins)
4. Infinite-dimensional chess (Bolan-Tsevas)
5. Tree embedding for ordinal positions
6. Large ordinals up to continuum (2^ℵ₀)
7. Neural network architecture
8. Engine evaluation
"""

import sys
sys.path.insert(0, '.')

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ██████╗ ██████╗ ██████╗ ██╗███╗   ██╗ █████╗ ██╗      ██████╗██╗  ██╗     ║
║  ██╔═══██╗██╔══██╗██╔══██╗██║████╗  ██║██╔══██╗██║     ██╔════╝██║  ██║     ║
║  ██║   ██║██████╔╝██║  ██║██║██╔██╗ ██║███████║██║     ██║     ███████║     ║
║  ██║   ██║██╔══██╗██║  ██║██║██║╚██╗██║██╔══██║██║     ██║     ██╔══██║     ║
║  ╚██████╔╝██║  ██║██████╔╝██║██║ ╚████║██║  ██║███████╗╚██████╗██║  ██║     ║
║   ╚═════╝ ╚═╝  ╚═╝╚═════╝ ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝ ╚═════╝╚═╝  ╚═╝     ║
║                                                                              ║
║           Transfinite Game Values Meet Neural Chess                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: TRANSFINITE ORDINALS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 80)
print("  SECTION 1: TRANSFINITE ORDINALS")
print("═" * 80)

from src.ordinals import OrdinalValue, OrdinalTier, OrdinalBucketizer

print("""
Ordinals extend the natural numbers into the transfinite:
  0, 1, 2, 3, ... → ω → ω+1 → ... → ω·2 → ... → ω² → ... → ω³ → ...

In chess, the ordinal game value represents "strategic depth" - how many
optimal moves until checkmate, including transfinite values for infinite games.
""")

# Create various ordinals
ordinals = [
    ("Mate in 1", OrdinalValue.from_ply(1)),
    ("Mate in 10", OrdinalValue.from_ply(10)),
    ("Mate in 100", OrdinalValue.from_ply(100)),
    ("ω (omega)", OrdinalValue.from_omega(1, 0)),
    ("ω + 5", OrdinalValue.from_omega(1, 5)),
    ("ω · 3", OrdinalValue.from_omega(3, 0)),
    ("ω²", OrdinalValue.from_omega_squared(1, 0, 0)),
    ("ω² + ω + 7", OrdinalValue.from_omega_squared(1, 1, 7)),
    ("ω³", OrdinalValue.from_omega_cubed(1, 0, 0, 0)),
    ("Draw", OrdinalValue.draw()),
]

print("  Ordinal Values and Their Tiers:")
print("  " + "-" * 50)
for name, ordinal in ordinals:
    tier = ordinal.tier()
    print(f"  {name:20} = {str(ordinal):15} (tier: {tier.name})")

# Demonstrate ordering
print("\n  Ordinal Ordering (< is well-defined):")
print("  " + "-" * 50)
comparisons = [
    (OrdinalValue.from_ply(1000000), OrdinalValue.from_omega(1, 0), "1,000,000 < ω"),
    (OrdinalValue.from_omega(1, 0), OrdinalValue.from_omega_squared(1, 0, 0), "ω < ω²"),
    (OrdinalValue.from_omega_squared(1, 0, 0), OrdinalValue.from_omega_cubed(1, 0, 0, 0), "ω² < ω³"),
]
for o1, o2, desc in comparisons:
    result = "✓" if o1 < o2 else "✗"
    print(f"  {result} {desc}")

# Bucketizer for neural network
print("\n  Ordinal Bucketizer (for neural network training):")
print("  " + "-" * 50)
bucketizer = OrdinalBucketizer()
print(f"  Total buckets: {bucketizer.n_buckets}")
print(f"  Finite ply buckets: 0-{bucketizer.max_finite_ply}")
print(f"  ω + k buckets: {bucketizer.n_omega_plus}")
print(f"  ω · n buckets: {bucketizer.n_omega_times}")
print(f"  ω² buckets: {bucketizer.n_omega_sq}")
print(f"  ω³ buckets: {bucketizer.n_omega_cube}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: STANDARD CHESS BOARD
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 80)
print("  SECTION 2: STANDARD 8×8 CHESS")
print("═" * 80)

from src.board import SparseBoard, BoardEncoder, Piece, PieceType, Color

print("\n  Standard Starting Position:")
board = SparseBoard.starting_position()
print(board)

print("\n  Board Encoding for Transformer:")
encoder = BoardEncoder(window_size=8)
encoding = encoder.encode_position(board)
print(f"  Token tensor shape: {encoding['tokens'].shape}")
print(f"  Position IDs shape: {encoding['position_ids'].shape}")
print(f"  Number of pieces: {len(encoding['piece_list'])}")

# Show piece tokens
print("\n  Piece Token Mapping:")
print("  " + "-" * 40)
token_map = {0: "Empty", 1: "♙ Pawn", 2: "♘ Knight", 3: "♗ Bishop",
             4: "♖ Rook", 5: "♕ Queen", 6: "♔ King",
             7: "♟ Pawn", 8: "♞ Knight", 9: "♝ Bishop",
             10: "♜ Rook", 11: "♛ Queen", 12: "♚ King"}
for i in range(0, 13, 2):
    left = f"{i}: {token_map[i]}"
    right = f"{i+1}: {token_map[i+1]}" if i+1 < 13 else ""
    print(f"  {left:20} {right}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: EXTENDED BOARDS (EVANS-HAMKINS)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 80)
print("  SECTION 3: EXTENDED BOARDS (Evans-Hamkins, arXiv 1302.4377)")
print("═" * 80)

print("""
Evans & Hamkins (2014) proved that 3D infinite chess achieves ALL countable
ordinal game values. They constructed positions with values ω, ω², ω³, etc.
""")

print("\n  Omega Ladder Position (game value = ω):")
print("  " + "-" * 50)
print("  Black king can retreat indefinitely, but White eventually wins.")
omega_board = SparseBoard.omega_ladder(12)
print(omega_board)

print("\n  Omega² Position (nested infinite structure):")
omega2_board = SparseBoard.omega_squared_position()
print(f"  Board bounds: {omega2_board.bounds}")
print(f"  Number of pieces: {len(omega2_board.pieces)}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: INFINITE-DIMENSIONAL CHESS (BOLAN-TSEVAS)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 80)
print("  SECTION 4: INFINITE-DIMENSIONAL CHESS (Bolan-Tsevas, 2024)")
print("═" * 80)

from src.dimensions import (
    Coordinate, InfiniteDimensionalBoard, BoardConfig, BoardVariant
)

print("""
Bolan & Tsevas (2024) proved that infinite-dimensional chess achieves
game values up to the CONTINUUM (2^ℵ₀) - the cardinality of real numbers!

Key insight: With infinitely many dimensions, there are continuum-many
possible moves, allowing construction of trees with continuum-sized ranks.
""")

print("\n  Coordinate System (Sparse Representation):")
print("  " + "-" * 50)

coords = [
    Coordinate.from_list([1, 2, 3]),
    Coordinate.from_list([0, 0, 0, 0, 5]),
    Coordinate.from_list([1, -1, 1, -1, 1, -1, 1]),
    Coordinate.from_list([3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]),
]

for c in coords:
    print(f"  {c}")
    print(f"    ||·||∞ = {c.sup_norm}, dims used: {sorted(c.dimensions_used)}")

print("\n  Supremum Norm Distance:")
print("  " + "-" * 50)
c1 = Coordinate.from_list([1, 2, 3])
c2 = Coordinate.from_list([2, 1, 2])
print(f"  c1 = {c1}")
print(f"  c2 = {c2}")
print(f"  d(c1, c2) = ||c1 - c2||∞ = {c1.distance(c2)}")
print(f"  King can move between them: {'Yes' if c1.distance(c2) <= 1 else 'No'}")

print("\n  Board Variants:")
print("  " + "-" * 50)
configs = [
    ("8×8×8×... (coords in [-4, 3])", BoardConfig.infinite_8()),
    ("3×3×3×... (coords in [-1, 1])", BoardConfig.infinite_3()),
    ("Weak pieces (finite coord changes)", BoardConfig.weak_pieces()),
]
for name, config in configs:
    print(f"  {name}")
    print(f"    Sidelength: {config.sidelength}, Weak pieces: {config.weak_pieces}")

print("\n  Infinite-Dimensional Board Demo:")
print("  " + "-" * 50)
inf_board = InfiniteDimensionalBoard(BoardConfig.infinite_8())

# Place pieces in different dimensions
inf_board.set_piece(Coordinate.from_list([0, 0]), Piece(PieceType.KING, Color.WHITE))
inf_board.set_piece(Coordinate.from_list([1, 1]), Piece(PieceType.ROOK, Color.WHITE))
inf_board.set_piece(Coordinate.from_list([2, 0]), Piece(PieceType.ROOK, Color.WHITE))
inf_board.set_piece(Coordinate.from_list([0, 0, 0, 0, 3]), Piece(PieceType.KING, Color.BLACK))

print(f"  White King at: {inf_board.king_positions[Color.WHITE]}")
print(f"  Black King at: {inf_board.king_positions[Color.BLACK]}")
print(f"  Distance between kings: {inf_board.king_positions[Color.WHITE].distance(inf_board.king_positions[Color.BLACK])}")

print("\n  2D Slice of Board (dimensions 0,1):")
print(inf_board.display_slice(dims=(0, 1), size=8))

print("\n  King Move Generation:")
white_king_coord = inf_board.king_positions[Color.WHITE]
king_moves = list(inf_board.generate_moves(white_king_coord))
print(f"  White king has {len(king_moves)} possible moves")
print(f"  Sample targets: {[m.to_coord for m in king_moves[:5]]}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: TREE EMBEDDING
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 80)
print("  SECTION 5: TREE EMBEDDING FOR ORDINAL POSITIONS")
print("═" * 80)

from src.tree_embedding import TreeNode, TreeBuilder, TreeEmbedder, OrdinalPositionGenerator

print("""
The Bolan-Tsevas construction embeds well-founded trees into chess positions.
The game value equals the ordinal rank of the tree:
  - Leaves have rank 0
  - rank(node) = sup{rank(child) + 1 : child in children}
""")

print("\n  Tree Examples:")
print("  " + "-" * 50)

# Linear tree
linear = TreeBuilder.linear_tree(5)
print(f"  Linear tree (depth 5):")
print(f"    Nodes: {sum(1 for _ in linear.all_nodes())}")
print(f"    Rank: {linear.rank}")

# Binary tree
binary = TreeBuilder.binary_tree(3)
print(f"\n  Binary tree (depth 3):")
print(f"    Nodes: {sum(1 for _ in binary.all_nodes())}")
print(f"    Leaves: {sum(1 for _ in binary.all_leaves())}")

# Omega power trees
for power in [1, 2, 3]:
    tree = TreeBuilder.omega_power_tree(power)
    print(f"\n  ω^{power} tree:")
    print(f"    Root children: {len(tree.children)}")
    if tree.children:
        print(f"    Grandchildren of first child: {len(tree.children[0].children)}")

print("\n  Position Generation from Trees:")
print("  " + "-" * 50)
generator = OrdinalPositionGenerator()

test_ordinals = [
    OrdinalValue.from_ply(3),
    OrdinalValue.from_omega(1, 0),
    OrdinalValue.from_omega_squared(1, 0, 0),
]

for ordinal in test_ordinals:
    pos = generator.position_for_ordinal(ordinal)
    print(f"  Ordinal {ordinal}: position with {len(pos.pieces)} pieces")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: LARGE ORDINALS & CONTINUUM
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 80)
print("  SECTION 6: LARGE ORDINALS & CONTINUUM CARDINALITY")
print("═" * 80)

from src.large_ordinals import (
    LargeOrdinal, OrdinalArithmetic, GameValueTheory,
    CardinalityClass, ORDINAL_CONSTANTS
)

print("""
The Veblen hierarchy extends ordinals far beyond ω³:
  ω, ω^ω, ω^ω^ω, ... → ε₀ (epsilon-zero) → ... → Γ₀ → ...

Bolan-Tsevas prove: infinite-dimensional chess achieves values up to 2^ℵ₀!
""")

print("\n  Ordinal Hierarchy:")
print("  " + "-" * 50)

hierarchy = [
    LargeOrdinal.omega(),
    LargeOrdinal.omega_power(2),
    LargeOrdinal.omega_power(3),
    LargeOrdinal.omega_omega(),
    LargeOrdinal.omega_tower_n(3),
    LargeOrdinal.epsilon_0(),
    LargeOrdinal.continuum(),
]

for i, ordinal in enumerate(hierarchy):
    card = ordinal.cardinality.name
    print(f"  {i+1}. {str(ordinal):20} (cardinality: {card})")

print("\n  Ordinal Comparisons:")
print("  " + "-" * 50)
print(f"  ω < ω^ω: {LargeOrdinal.omega() < LargeOrdinal.omega_omega()}")
print(f"  ω^ω < ε₀: {LargeOrdinal.omega_omega() < LargeOrdinal.epsilon_0()}")
print(f"  ε₀ < continuum: {LargeOrdinal.epsilon_0() < LargeOrdinal.continuum()}")

print("\n  Ordinal Arithmetic:")
print("  " + "-" * 50)
omega = LargeOrdinal.omega()
result = OrdinalArithmetic.add(omega, LargeOrdinal.finite(5))
print(f"  ω + 5 = {result}")
result = OrdinalArithmetic.multiply(omega, omega)
print(f"  ω × ω = {result}")
result = OrdinalArithmetic.power(omega, omega)
print(f"  ω^ω = {result}")

print("\n  Game Value Theory (Bolan-Tsevas 2024):")
print("  " + "-" * 50)
print(f"  Maximum value (full pieces): {GameValueTheory.max_value_full_pieces()}")
print(f"  Maximum value (weak pieces): {GameValueTheory.max_value_weak_pieces()}")
print(f"  Moves per position (full): {GameValueTheory.moves_per_position_full()}")
print(f"  Moves per position (weak): {GameValueTheory.moves_per_position_weak()}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: NEURAL NETWORK ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 80)
print("  SECTION 7: TRANSFORMER ARCHITECTURE")
print("═" * 80)

import torch
from src.transformer import OrdinalChessTransformer, TransformerConfig

print("""
Based on Google DeepMind's searchless_chess (NeurIPS 2024), we use a
decoder-only transformer with multiple prediction heads:

  ┌─────────────────────────────────────────────────┐
  │           Transformer Backbone                   │
  │  (Piece Embeddings + Relative Position + Turn)  │
  └─────────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ↓               ↓               ↓
  ┌──────────┐   ┌──────────┐   ┌──────────────┐
  │  Value   │   │  Policy  │   │   Ordinal    │
  │  Head    │   │   Head   │   │    Head      │
  │ (W/D/L)  │   │ (moves)  │   │  (buckets)   │
  └──────────┘   └──────────┘   └──────────────┘
""")

print("\n  Model Configurations:")
print("  " + "-" * 50)

for name, config in [("Small", TransformerConfig.small()),
                      ("Medium", TransformerConfig.medium()),
                      ("Large", TransformerConfig.large())]:
    params = config.n_embed * config.n_embed * config.n_layers * 12 // 1e6
    print(f"  {name:8}: {config.n_embed}d, {config.n_heads} heads, {config.n_layers} layers (~{params:.0f}M params)")

print("\n  Creating Small Model...")
config = TransformerConfig.small()
model = OrdinalChessTransformer(config)

print("\n  Forward Pass Demo:")
print("  " + "-" * 50)
batch_size = 1
tokens = torch.randint(0, 13, (batch_size, config.window_size, config.window_size))
rel_pos = torch.randint(-8, 8, (batch_size, config.window_size, config.window_size, 2))
turn = torch.tensor([0])  # White to move

with torch.no_grad():
    outputs = model(tokens, rel_pos, turn)

print("  Output shapes:")
for key in ['value', 'policy', 'ordinal_bucket_probs', 'ordinal_predicted_bucket']:
    if key in outputs:
        print(f"    {key}: {outputs[key].shape}")

# Show predictions
value_probs = outputs['value'][0].numpy()
print(f"\n  Value prediction: Loss={value_probs[0]:.3f}, Draw={value_probs[1]:.3f}, Win={value_probs[2]:.3f}")

ordinal_bucket = outputs['ordinal_predicted_bucket'][0].item()
predicted_ordinal = model.bucketizer.from_bucket(ordinal_bucket)
print(f"  Ordinal prediction: bucket {ordinal_bucket} → {predicted_ordinal} ({predicted_ordinal.tier().name})")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: CHESS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 80)
print("  SECTION 8: ORDINALCHESS ENGINE")
print("═" * 80)

from src.engine import OrdinalChessEngine, EngineConfig

print("""
The OrdinalChess engine provides:
  - Move selection via transformer evaluation
  - Ordinal game value predictions
  - UCI protocol compatibility
  - Both standard and extended board support
""")

print("\n  Creating Engine...")
engine = OrdinalChessEngine()
engine.new_game()

print("\n  Starting Position Evaluation:")
print("  " + "-" * 50)
evaluation = engine.evaluate()

print(f"  Best move: {evaluation.best_move}")
print(f"  Ordinal value: {evaluation.ordinal_value}")
print(f"  Ordinal tier: {evaluation.ordinal_value.tier().name}")
print(f"  Win probability: {evaluation.win_probability:.3f}")
print(f"  Draw probability: {evaluation.draw_probability:.3f}")
print(f"  Think time: {evaluation.think_time_ms}ms")

print("\n  After 1.e4 e5:")
engine.set_position("startpos", ["e2e4", "e7e5"])
evaluation = engine.evaluate()
print(f"  Best move: {evaluation.best_move}")
print(f"  Ordinal value: {evaluation.ordinal_value}")
print(f"  Win probability: {evaluation.win_probability:.3f}")

print("\n  Top 5 Moves:")
for move, score in evaluation.top_moves[:5]:
    print(f"    {move}: {score:.3f}")

print("\n  UCI Info String:")
print(f"  {evaluation.to_uci_info()}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 80)
print("  SECTION 9: TRAINING DATA GENERATION")
print("═" * 80)

from src.data import OrdinalMotifGenerator, OrdinalChessDataset, analyze_dataset

print("""
Synthetic positions with known ordinal values for training:
  - Mate-in-N positions (finite ordinals)
  - Omega ladder positions (ω, ω+k)
  - Nested structures (ω·n, ω², ω³)
""")

print("\n  Motif Generator Examples:")
print("  " + "-" * 50)
generator = OrdinalMotifGenerator(seed=42)

samples = [
    ("Finite (mate in 5)", generator.generate_finite_mate(5)),
    ("Omega ladder", generator.generate_omega_ladder(0)),
    ("Omega + 7", generator.generate_omega_ladder(2)),
    ("Omega × 3", generator.generate_omega_times_n(3)),
    ("Omega squared", generator.generate_omega_squared()),
    ("Omega cubed", generator.generate_omega_cubed()),
    ("Draw", generator.generate_draw()),
]

for name, sample in samples:
    print(f"  {name:20}: ordinal = {sample.ordinal}")

print("\n  Dataset Statistics (1000 samples):")
print("  " + "-" * 50)
dataset = OrdinalChessDataset(n_samples=1000, seed=42)
stats = analyze_dataset(dataset)

total = sum(stats.values())
for tier, count in sorted(stats.items(), key=lambda x: -x[1]):
    if count > 0:
        bar = "█" * int(count / total * 40)
        print(f"  {tier:15}: {count:4} ({100*count/total:5.1f}%) {bar}")


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 80)
print("  SUMMARY: ORDINALCHESS CAPABILITIES")
print("═" * 80)

print("""
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  ORDINALCHESS: Transfinite Game Values Meet Neural Chess                │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                                                                         │
  │  THEORETICAL FOUNDATIONS:                                               │
  │    • searchless_chess (DeepMind, NeurIPS 2024) - 2895 Elo transformers │
  │    • Evans-Hamkins (2014) - Countable ordinals in 3D infinite chess    │
  │    • Bolan-Tsevas (2024) - Continuum ordinals in infinite dimensions   │
  │                                                                         │
  │  GAME VALUE HIERARCHY:                                                  │
  │    Standard Chess    → Finite ordinals (mate in N)                     │
  │    Extended/3D       → Countable ordinals (ω, ω², ω³, ..., ω₁)        │
  │    Infinite-Dim      → Continuum ordinals (up to 2^ℵ₀)                 │
  │    Weak Pieces       → Countable ordinals only                         │
  │                                                                         │
  │  NEURAL ARCHITECTURE:                                                   │
  │    • Decoder-only transformer (9M - 270M parameters)                   │
  │    • Relative position encoding (supports infinite boards)             │
  │    • Multi-head output: Value, Policy, Action-Value, Ordinal           │
  │    • Cumulative ordinal regression for proper ordering                 │
  │                                                                         │
  │  UNIQUE FEATURES:                                                       │
  │    • First chess engine predicting transfinite game values             │
  │    • Supremum norm geometry for infinite dimensions                    │
  │    • Tree embedding for constructing positions with target ordinals    │
  │    • Full UCI protocol support with ordinal extensions                 │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘

  Repository: https://github.com/yurekami/ordinal_chess

  "Where ω is just the beginning."
""")

print("═" * 80)
print("  DEMO COMPLETE")
print("═" * 80 + "\n")
