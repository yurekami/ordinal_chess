# OrdinalChess: Transfinite Game Values Meet Neural Chess

> **A novel synthesis of Google DeepMind's searchless_chess and Evans-Hamkins infinite chess theory**

OrdinalChess is an innovative chess AI that predicts not just win probability, but the **transfinite ordinal game value** of positions. This provides unprecedented insight into the strategic depth of chess positions, distinguishing between positions that are "barely winning" and those with deep, forcing advantages.

## Overview

### The Insight

Traditional chess engines evaluate positions with a single scalar (centipawns). But game theory tells us positions have richer structure:

- **Mate in 5** has game value **5** (finite ordinal)
- **Forced win with infinite escape sequences** has value **ω** (omega)
- **Nested infinite structures** have values like **ω²**, **ω³**, etc.

OrdinalChess brings this mathematical richness to practical chess AI.

### Sources

1. **[searchless_chess](https://github.com/google-deepmind/searchless_chess)** (NeurIPS 2024)
   - 270M parameter transformer achieving 2895 Elo
   - Pure pattern recognition without tree search
   - Demonstrates neural networks can learn strong chess

2. **[Transfinite Game Values in Infinite Chess](https://arxiv.org/abs/1302.4377)** (Evans & Hamkins, 2013)
   - Mathematical framework for ordinal game values
   - Proves infinite chess can achieve values ω, ω², ω³, ...
   - Foundation for understanding strategic depth

## Features

### Ordinal Value Prediction

```python
from ordinal_chess import OrdinalChessEngine, OrdinalValue

engine = OrdinalChessEngine()
engine.set_position("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")

evaluation = engine.evaluate()
print(f"Ordinal value: {evaluation.ordinal_value}")  # e.g., "42" or "ω + 3"
print(f"Ordinal tier: {evaluation.ordinal_value.tier().name}")  # FINITE, OMEGA, OMEGA_SQUARED, etc.
```

### Extended Board Support

```python
from ordinal_chess import SparseBoard

# Standard 8x8 board
board = SparseBoard.starting_position()

# Omega ladder position (infinite retreat pattern)
omega_board = SparseBoard.omega_ladder(n=20)
print(f"Game value: ω")  # White wins, but Black can delay infinitely

# Custom extended board
extended = SparseBoard(bounds=(0, 0, 31, 31), is_infinite=False)
```

### Multi-Head Transformer Architecture

```python
from ordinal_chess import OrdinalChessTransformer, TransformerConfig

# Configure model size
config = TransformerConfig.large()  # 270M parameters

model = OrdinalChessTransformer(config)
# Outputs:
# - value: Win/Draw/Loss probabilities
# - action_values: Q-values for each move
# - policy: Move probability distribution
# - ordinal_bucket_probs: Distribution over ordinal values
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OrdinalChess Transformer                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐   ┌──────────────┐   ┌─────────────────┐  │
│  │   Piece     │ + │   Relative   │ + │      Turn       │  │
│  │  Embeddings │   │   Position   │   │    Embedding    │  │
│  └─────────────┘   └──────────────┘   └─────────────────┘  │
│           │                │                   │            │
│           └────────────────┴───────────────────┘            │
│                            │                                 │
│                    ┌───────▼───────┐                        │
│                    │   [CLS] + Seq │                        │
│                    └───────────────┘                        │
│                            │                                 │
│                    ┌───────▼───────┐                        │
│                    │  Transformer  │ × N layers             │
│                    │    Blocks     │                        │
│                    └───────────────┘                        │
│                            │                                 │
│         ┌──────────┬───────┴───────┬──────────┐            │
│         ▼          ▼               ▼          ▼            │
│  ┌────────────┐ ┌────────┐ ┌────────────┐ ┌────────────┐  │
│  │   Value    │ │ Policy │ │   Action   │ │  Ordinal   │  │
│  │   Head     │ │  Head  │ │   Values   │ │   Head     │  │
│  │ (WDL prob) │ │ (moves)│ │  (Q-vals)  │ │ (buckets)  │  │
│  └────────────┘ └────────┘ └────────────┘ └────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Ordinal Value Theory

### Cantor Normal Form

Ordinals are represented in Cantor Normal Form:

```
α = ω^β₁·n₁ + ω^β₂·n₂ + ... + ω^βₖ·nₖ
```

where β₁ > β₂ > ... > βₖ and nᵢ are positive integers.

### Examples

| Position Type | Ordinal Value | Meaning |
|--------------|---------------|---------|
| Mate in 3 | 3 | White mates in 3 moves |
| Mate in 100 | 100 | White mates in 100 moves |
| Omega ladder | ω | White wins, infinite delay possible |
| Omega + 5 | ω + 5 | Omega structure plus 5 forced moves |
| Double ladder | ω·2 | Two sequential omega structures |
| Nested infinite | ω² | Omega-many omega structures |
| Triple nested | ω³ | Deeply nested forcing tree |

### Tier Classification

```python
class OrdinalTier(IntEnum):
    FINITE = 0          # Standard finite ply (0-500+)
    OMEGA = 1           # ω: basic infinite
    OMEGA_PLUS = 2      # ω + n: omega plus finite
    OMEGA_TIMES = 3     # ω · n: omega times finite
    OMEGA_SQUARED = 4   # ω²: omega squared
    OMEGA_SQ_PLUS = 5   # ω² + lower terms
    OMEGA_CUBED = 6     # ω³: omega cubed
    OMEGA_HIGHER = 7    # ω^k for k > 3
```

## Training

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train small model
python -m ordinal_chess.train

# Train with custom config
python -m ordinal_chess.train --model-size medium --max-steps 50000
```

### Training Pipeline

1. **Data Generation**: Synthetic positions with known ordinal values
2. **Multi-head Loss**: Combined value, policy, and ordinal objectives
3. **Curriculum Learning**: Gradual introduction of transfinite positions
4. **Evaluation**: Puzzle solving, ordinal accuracy, Kendall's tau

## Evaluation

### Metrics

- **Bucket Accuracy**: Exact ordinal bucket prediction
- **Tier Accuracy**: Correct ordinal tier classification
- **Kendall's Tau**: Ordering correlation between predicted and true ordinals
- **Move Accuracy**: Correct move selection on puzzles
- **Elo Rating**: Tournament performance against other engines

### Running Evaluation

```python
from ordinal_chess.evaluation import run_full_evaluation

engine = OrdinalChessEngine()
results = run_full_evaluation(engine)
print(f"Tier accuracy: {results['ordinal']['tier_accuracy']:.2%}")
```

## UCI Interface

OrdinalChess implements the UCI protocol with extensions:

```
position startpos moves e2e4 e7e5
go movetime 1000

# Output:
info depth 1 score cp 35 pv d2d4 string ordinal=42 tier=FINITE
bestmove d2d4

# Custom eval command:
eval
# Output:
Position evaluation:
  Ordinal value: 42
  Ordinal tier: FINITE
  Win/Draw/Loss: 0.550/0.300/0.150
  Best move: d2d4
```

## Project Structure

```
ordinal_chess/
├── src/
│   ├── __init__.py      # Package exports
│   ├── ordinals.py      # Transfinite ordinal representation
│   ├── board.py         # Extended/infinite board support
│   ├── transformer.py   # Neural network architecture
│   ├── data.py          # Dataset and data generation
│   ├── train.py         # Training pipeline
│   ├── engine.py        # Chess engine interface
│   └── evaluation.py    # Evaluation and tournaments
├── tests/               # Unit tests
├── configs/             # Training configurations
├── checkpoints/         # Model checkpoints
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## References

1. **Searchless Chess** - Ruoss, A., et al. (2024). "Grandmaster-Level Chess Without Search." NeurIPS 2024. [GitHub](https://github.com/google-deepmind/searchless_chess)

2. **Transfinite Game Values** - Evans, C.D.A. & Hamkins, J.D. (2014). "Transfinite game values in infinite chess." [arXiv:1302.4377](https://arxiv.org/abs/1302.4377)

3. **MAKER Framework** - Multi-agent coordination and ensemble methods for robust AI systems.

## Citation

```bibtex
@software{ordinalchess2025,
  title={OrdinalChess: Transfinite Game Values Meet Neural Chess},
  author={OrdinalChess Team},
  year={2025},
  url={https://github.com/ordinalchess/ordinalchess}
}
```

## License

MIT License - See LICENSE file for details.

---

*OrdinalChess: Where ω is just the beginning.*
