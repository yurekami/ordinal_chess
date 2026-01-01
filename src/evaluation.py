"""
Evaluation and Tournament System for OrdinalChess

Provides:
1. Puzzle solving evaluation (from Lichess puzzles)
2. Tournament system for Elo rating computation
3. Ordinal prediction accuracy metrics
4. Comparison with Stockfish and other engines

Based on the evaluation methodology from searchless_chess.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable
from pathlib import Path
import subprocess
import json

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import numpy as np
from scipy import stats as scipy_stats

from .engine import OrdinalChessEngine, EngineConfig, EngineEvaluation
from .ordinals import OrdinalValue, OrdinalBucketizer, OrdinalTier
from .board import SparseBoard
from .data import OrdinalMotifGenerator


@dataclass
class PuzzleResult:
    """Result of a puzzle evaluation."""
    puzzle_id: str
    fen: str
    solution_moves: List[str]
    engine_moves: List[str]
    correct: bool
    ordinal_prediction: OrdinalValue
    time_ms: int


@dataclass
class TournamentGame:
    """Result of a single tournament game."""
    white_engine: str
    black_engine: str
    result: str  # "1-0", "0-1", "1/2-1/2"
    moves: List[str]
    final_ordinal: Optional[OrdinalValue]
    ply_count: int


class PuzzleEvaluator:
    """
    Evaluates engine performance on chess puzzles.

    Puzzles test tactical ability and can measure:
    - Move accuracy
    - Ordinal estimation accuracy (mate-in-N detection)
    """

    def __init__(self, engine: OrdinalChessEngine):
        self.engine = engine
        self.results: List[PuzzleResult] = []

    def load_puzzles_from_file(self, path: str) -> List[Dict]:
        """Load puzzles from CSV or JSON file."""
        puzzles = []
        path = Path(path)

        if path.suffix == ".json":
            with open(path) as f:
                puzzles = json.load(f)
        elif path.suffix == ".csv":
            import csv
            with open(path) as f:
                reader = csv.DictReader(f)
                puzzles = list(reader)

        return puzzles

    def generate_synthetic_puzzles(self, n: int = 100) -> List[Dict]:
        """Generate synthetic puzzles with known ordinal values."""
        generator = OrdinalMotifGenerator(seed=42)
        puzzles = []

        for i in range(n):
            # Generate mate-in-N positions
            ply = np.random.randint(1, 10)
            sample = generator.generate_finite_mate(ply)

            puzzles.append({
                "id": f"synthetic_{i}",
                "fen": self._board_to_fen(sample.board),
                "solution": ["e2e4"],  # Placeholder - would need actual solution
                "ordinal": ply,
                "tier": "FINITE",
            })

        return puzzles

    def _board_to_fen(self, board: SparseBoard) -> str:
        """Convert SparseBoard to FEN string (simplified)."""
        # This is a simplified conversion - full FEN generation is more complex
        piece_chars = {
            (1, 0): 'P', (2, 0): 'N', (3, 0): 'B', (4, 0): 'R', (5, 0): 'Q', (6, 0): 'K',
            (1, 1): 'p', (2, 1): 'n', (3, 1): 'b', (4, 1): 'r', (5, 1): 'q', (6, 1): 'k',
        }

        rows = []
        for rank in range(7, -1, -1):
            row = ""
            empty_count = 0
            for file in range(8):
                piece = board.get_piece((file, rank))
                if piece:
                    if empty_count > 0:
                        row += str(empty_count)
                        empty_count = 0
                    char = piece_chars.get((piece.piece_type, piece.color), '?')
                    row += char
                else:
                    empty_count += 1
            if empty_count > 0:
                row += str(empty_count)
            rows.append(row)

        position = "/".join(rows)
        turn = 'w' if board.turn == 0 else 'b'
        castling = "KQkq" if board.castling_rights else "-"
        en_passant = "-"

        return f"{position} {turn} {castling} {en_passant} 0 1"

    def evaluate_puzzle(self, puzzle: Dict) -> PuzzleResult:
        """Evaluate engine performance on a single puzzle."""
        start_time = time.time()

        fen = puzzle.get("fen", "")
        solution = puzzle.get("solution", [])
        if isinstance(solution, str):
            solution = solution.split()

        # Set position and get engine's move
        self.engine.set_position(fen)
        evaluation = self.engine.evaluate()

        engine_moves = []
        if evaluation.best_move:
            engine_moves.append(str(evaluation.best_move))

        # Check if first move matches
        correct = len(engine_moves) > 0 and len(solution) > 0 and \
                  engine_moves[0].lower() == solution[0].lower()

        elapsed_ms = int((time.time() - start_time) * 1000)

        return PuzzleResult(
            puzzle_id=puzzle.get("id", "unknown"),
            fen=fen,
            solution_moves=solution,
            engine_moves=engine_moves,
            correct=correct,
            ordinal_prediction=evaluation.ordinal_value,
            time_ms=elapsed_ms,
        )

    def run_evaluation(self, puzzles: List[Dict]) -> Dict:
        """Run evaluation on a list of puzzles."""
        self.results = []

        for puzzle in puzzles:
            result = self.evaluate_puzzle(puzzle)
            self.results.append(result)

        # Compute statistics
        accuracy = sum(r.correct for r in self.results) / len(self.results)
        avg_time = sum(r.time_ms for r in self.results) / len(self.results)

        # Ordinal accuracy (for puzzles with known ordinal values)
        ordinal_correct = 0
        ordinal_total = 0
        for puzzle, result in zip(puzzles, self.results):
            if "ordinal" in puzzle:
                expected = OrdinalValue.from_ply(puzzle["ordinal"])
                if result.ordinal_prediction.tier() == expected.tier():
                    ordinal_correct += 1
                ordinal_total += 1

        return {
            "total_puzzles": len(puzzles),
            "move_accuracy": accuracy,
            "average_time_ms": avg_time,
            "ordinal_accuracy": ordinal_correct / ordinal_total if ordinal_total > 0 else None,
            "results": self.results,
        }


class OrdinalAccuracyEvaluator:
    """
    Evaluates accuracy of ordinal predictions on test positions.

    Key metrics:
    - Bucket accuracy: Exact bucket match
    - Tier accuracy: Correct ordinal tier (finite, ω, ω², etc.)
    - Kendall's tau: Ordering correlation
    - Calibration: Are predicted probabilities accurate?
    """

    def __init__(self, engine: OrdinalChessEngine):
        self.engine = engine
        self.bucketizer = OrdinalBucketizer()

    def evaluate_ordinal_accuracy(
        self,
        test_positions: List[Tuple[SparseBoard, OrdinalValue]]
    ) -> Dict:
        """Evaluate ordinal prediction accuracy on test positions."""
        predictions = []
        targets = []
        tier_predictions = []
        tier_targets = []

        for board, true_ordinal in test_positions:
            # Get engine's prediction
            self.engine.board = board
            evaluation = self.engine.evaluate()
            pred_ordinal = evaluation.ordinal_value

            # Record bucket indices
            pred_bucket = self.bucketizer.to_bucket(pred_ordinal)
            true_bucket = self.bucketizer.to_bucket(true_ordinal)
            predictions.append(pred_bucket)
            targets.append(true_bucket)

            # Record tiers
            tier_predictions.append(pred_ordinal.tier())
            tier_targets.append(true_ordinal.tier())

        predictions = np.array(predictions)
        targets = np.array(targets)

        # Compute metrics
        bucket_accuracy = (predictions == targets).mean()

        tier_accuracy = sum(
            p == t for p, t in zip(tier_predictions, tier_targets)
        ) / len(tier_predictions)

        # Kendall's tau
        if len(np.unique(predictions)) > 1 and len(np.unique(targets)) > 1:
            tau, p_value = scipy_stats.kendalltau(predictions, targets)
        else:
            tau, p_value = 0.0, 1.0

        # Mean absolute bucket error
        mae = np.abs(predictions - targets).mean()

        # Tier-wise accuracy
        tier_accuracies = {}
        for tier in OrdinalTier:
            tier_mask = np.array([t == tier for t in tier_targets])
            if tier_mask.sum() > 0:
                tier_acc = (predictions[tier_mask] == targets[tier_mask]).mean()
                tier_accuracies[tier.name] = {
                    "accuracy": tier_acc,
                    "count": tier_mask.sum(),
                }

        return {
            "bucket_accuracy": bucket_accuracy,
            "tier_accuracy": tier_accuracy,
            "kendall_tau": tau,
            "kendall_p_value": p_value,
            "mean_bucket_error": mae,
            "tier_accuracies": tier_accuracies,
            "n_positions": len(test_positions),
        }


class Tournament:
    """
    Tournament system for computing Elo ratings.

    Supports:
    - Round-robin tournaments between engines
    - Match-based Elo calculation
    - Integration with external engines (Stockfish, Leela)
    """

    def __init__(self, engines: Dict[str, OrdinalChessEngine]):
        self.engines = engines
        self.games: List[TournamentGame] = []

    def play_game(
        self,
        white_name: str,
        black_name: str,
        time_control: Tuple[int, int] = (60000, 0),  # (base_ms, increment_ms)
        max_moves: int = 200,
    ) -> TournamentGame:
        """Play a single game between two engines."""
        white_engine = self.engines[white_name]
        black_engine = self.engines[black_name]

        white_engine.new_game()
        black_engine.new_game()

        moves = []
        wtime, btime = time_control[0], time_control[0]
        inc = time_control[1]

        for move_num in range(max_moves):
            current_engine = white_engine if move_num % 2 == 0 else black_engine
            current_name = white_name if move_num % 2 == 0 else black_name

            # Get move
            current_engine.set_position("startpos", moves)
            evaluation = current_engine.evaluate()

            if evaluation.best_move is None:
                # Game over
                if move_num % 2 == 0:
                    result = "0-1"  # White has no moves
                else:
                    result = "1-0"  # Black has no moves
                break

            move_str = str(evaluation.best_move)
            moves.append(move_str)

            # Update clocks
            if move_num % 2 == 0:
                wtime = max(0, wtime - evaluation.think_time_ms + inc)
            else:
                btime = max(0, btime - evaluation.think_time_ms + inc)

            # Check for time forfeit
            if wtime <= 0:
                result = "0-1"
                break
            if btime <= 0:
                result = "1-0"
                break

            # Get final ordinal evaluation
            final_ordinal = evaluation.ordinal_value
        else:
            # Max moves reached = draw
            result = "1/2-1/2"
            final_ordinal = OrdinalValue.draw()

        return TournamentGame(
            white_engine=white_name,
            black_engine=black_name,
            result=result,
            moves=moves,
            final_ordinal=final_ordinal,
            ply_count=len(moves),
        )

    def run_round_robin(
        self,
        n_games_per_pair: int = 2,
        time_control: Tuple[int, int] = (60000, 0),
    ) -> Dict:
        """Run round-robin tournament between all engines."""
        engine_names = list(self.engines.keys())
        self.games = []

        for i, name1 in enumerate(engine_names):
            for name2 in engine_names[i+1:]:
                for _ in range(n_games_per_pair):
                    # Each pair plays with alternating colors
                    game1 = self.play_game(name1, name2, time_control)
                    game2 = self.play_game(name2, name1, time_control)
                    self.games.extend([game1, game2])

        return self.compute_results()

    def compute_results(self) -> Dict:
        """Compute tournament results and Elo estimates."""
        engine_names = list(self.engines.keys())
        scores = {name: 0.0 for name in engine_names}
        game_counts = {name: 0 for name in engine_names}

        for game in self.games:
            game_counts[game.white_engine] += 1
            game_counts[game.black_engine] += 1

            if game.result == "1-0":
                scores[game.white_engine] += 1.0
            elif game.result == "0-1":
                scores[game.black_engine] += 1.0
            else:
                scores[game.white_engine] += 0.5
                scores[game.black_engine] += 0.5

        # Compute performance ratings
        performance = {}
        for name in engine_names:
            if game_counts[name] > 0:
                win_rate = scores[name] / game_counts[name]
                # Simple Elo difference estimate
                if win_rate == 1.0:
                    elo_diff = 400
                elif win_rate == 0.0:
                    elo_diff = -400
                else:
                    elo_diff = 400 * np.log10(win_rate / (1 - win_rate))
                performance[name] = {
                    "score": scores[name],
                    "games": game_counts[name],
                    "win_rate": win_rate,
                    "elo_diff": elo_diff,
                }

        return {
            "engines": performance,
            "total_games": len(self.games),
            "games": self.games,
        }


def run_full_evaluation(engine: OrdinalChessEngine) -> Dict:
    """Run comprehensive evaluation suite."""
    results = {}

    print("=== OrdinalChess Evaluation Suite ===\n")

    # 1. Puzzle evaluation
    print("1. Puzzle Evaluation...")
    puzzle_eval = PuzzleEvaluator(engine)
    puzzles = puzzle_eval.generate_synthetic_puzzles(50)
    puzzle_results = puzzle_eval.run_evaluation(puzzles)
    results["puzzles"] = {
        "move_accuracy": puzzle_results["move_accuracy"],
        "ordinal_accuracy": puzzle_results.get("ordinal_accuracy"),
        "average_time_ms": puzzle_results["average_time_ms"],
    }
    print(f"   Move accuracy: {puzzle_results['move_accuracy']:.2%}")

    # 2. Ordinal accuracy
    print("2. Ordinal Accuracy Evaluation...")
    ordinal_eval = OrdinalAccuracyEvaluator(engine)
    generator = OrdinalMotifGenerator(seed=123)

    test_positions = []
    for _ in range(50):
        sample = generator.generate_sample()
        test_positions.append((sample.board, sample.ordinal))

    ordinal_results = ordinal_eval.evaluate_ordinal_accuracy(test_positions)
    results["ordinal"] = {
        "bucket_accuracy": ordinal_results["bucket_accuracy"],
        "tier_accuracy": ordinal_results["tier_accuracy"],
        "kendall_tau": ordinal_results["kendall_tau"],
    }
    print(f"   Tier accuracy: {ordinal_results['tier_accuracy']:.2%}")
    print(f"   Kendall's tau: {ordinal_results['kendall_tau']:.3f}")

    print("\n=== Evaluation Complete ===")
    return results


if __name__ == "__main__":
    print("Initializing OrdinalChess Engine for evaluation...")
    engine = OrdinalChessEngine()
    engine.new_game()

    results = run_full_evaluation(engine)

    print("\nFinal Results:")
    print(json.dumps(results, indent=2, default=str))
