"""
OrdinalChess Engine Interface

Provides a chess engine that:
1. Uses the trained OrdinalChess transformer for move selection
2. Reports ordinal game values alongside standard evaluations
3. Supports both standard 8x8 and extended boards
4. Implements UCI-compatible interface for integration with chess GUIs

This engine uniquely provides transfinite ordinal evaluations, giving
insight into the "strategic depth" of positions beyond simple win/loss.
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Generator
from pathlib import Path

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import numpy as np

from .transformer import OrdinalChessTransformer, TransformerConfig
from .ordinals import OrdinalValue, OrdinalBucketizer, OrdinalTier
from .board import SparseBoard, BoardEncoder, Move, Coord, PieceType, Color, Piece


@dataclass
class EngineConfig:
    """Configuration for the OrdinalChess engine."""
    model_path: Optional[str] = None
    model_size: str = "small"
    device: str = "cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu"

    # Search settings (for hybrid search if desired)
    use_mcts: bool = False           # Optional MCTS layer
    mcts_simulations: int = 100
    temperature: float = 0.1         # Move selection temperature

    # Time management
    move_overhead_ms: int = 50
    min_think_time_ms: int = 100


@dataclass
class EngineEvaluation:
    """Complete engine evaluation of a position."""
    best_move: Optional[Move]
    ordinal_value: OrdinalValue
    win_probability: float            # P(win) from current player's perspective
    draw_probability: float
    loss_probability: float
    top_moves: List[Tuple[Move, float]]  # Top moves with action values
    think_time_ms: int
    nodes_searched: int = 1           # For UCI compatibility

    def to_uci_info(self) -> str:
        """Format evaluation as UCI info string."""
        # Convert ordinal to centipawn-like score
        if self.ordinal_value.is_transfinite():
            # Transfinite = very high score
            cp_score = 10000 + self.ordinal_value.tier().value * 1000
        elif self.ordinal_value.is_loss:
            cp_score = -32000
        elif self.ordinal_value.is_draw:
            cp_score = 0
        else:
            # Convert win probability to centipawns
            # Using logistic formula: cp = 400 * log10(p/(1-p))
            p = max(0.001, min(0.999, self.win_probability))
            cp_score = int(400 * np.log10(p / (1 - p)))

        info = f"info depth 1 nodes {self.nodes_searched} time {self.think_time_ms}"
        info += f" score cp {cp_score}"

        if self.best_move:
            info += f" pv {self.best_move}"

        # Add custom ordinal info
        info += f" string ordinal={self.ordinal_value} tier={self.ordinal_value.tier().name}"

        return info


class OrdinalChessEngine:
    """
    Main OrdinalChess engine class.

    Provides move selection and position evaluation using the
    trained OrdinalChess transformer model.
    """

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self.device = torch.device(self.config.device) if HAS_TORCH else None

        # Initialize model
        self._init_model()

        # Board encoder
        self.encoder = BoardEncoder(window_size=16)

        # Current position
        self.board: Optional[SparseBoard] = None
        self.history: List[Move] = []

    def _init_model(self):
        """Initialize or load the transformer model."""
        if not HAS_TORCH:
            print("Warning: PyTorch not available, engine will use random moves")
            self.model = None
            return

        # Create model
        model_configs = {
            "small": TransformerConfig.small(),
            "medium": TransformerConfig.medium(),
            "large": TransformerConfig.large(),
        }
        model_config = model_configs[self.config.model_size]
        self.model = OrdinalChessTransformer(model_config)

        # Load weights if path provided
        if self.config.model_path and Path(self.config.model_path).exists():
            checkpoint = torch.load(self.config.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {self.config.model_path}")

        self.model.to(self.device)
        self.model.eval()

    def new_game(self):
        """Start a new game with standard starting position."""
        self.board = SparseBoard.starting_position()
        self.history = []

    def set_position(self, fen: str = "startpos", moves: List[str] = None):
        """
        Set position from FEN or starting position with moves.

        Args:
            fen: FEN string or "startpos" for starting position
            moves: List of moves in UCI format to apply
        """
        if fen == "startpos":
            self.board = SparseBoard.starting_position()
        else:
            self.board = SparseBoard.from_fen(fen)

        self.history = []

        if moves:
            for move_str in moves:
                move = self._parse_uci_move(move_str)
                if move:
                    self._apply_move(move)
                    self.history.append(move)

    def _parse_uci_move(self, uci: str) -> Optional[Move]:
        """Parse UCI move string (e.g., 'e2e4', 'e7e8q')."""
        if len(uci) < 4:
            return None

        from_file = ord(uci[0]) - ord('a')
        from_rank = int(uci[1]) - 1
        to_file = ord(uci[2]) - ord('a')
        to_rank = int(uci[3]) - 1

        promotion = None
        if len(uci) > 4:
            promo_map = {'q': PieceType.QUEEN, 'r': PieceType.ROOK,
                        'b': PieceType.BISHOP, 'n': PieceType.KNIGHT}
            promotion = promo_map.get(uci[4].lower())

        return Move(
            from_square=(from_file, from_rank),
            to_square=(to_file, to_rank),
            promotion=promotion
        )

    def _apply_move(self, move: Move):
        """Apply a move to the current board."""
        if not self.board:
            return

        piece = self.board.get_piece(move.from_square)
        if not piece:
            return

        # Handle promotion
        if move.promotion:
            piece = Piece(move.promotion, piece.color)

        # Move the piece
        self.board.set_piece(move.from_square, None)
        self.board.set_piece(move.to_square, piece)

        # TODO: Handle castling, en passant, etc.

        # Switch turn
        self.board.turn = Color.BLACK if self.board.turn == Color.WHITE else Color.WHITE

    def _generate_legal_moves(self) -> List[Move]:
        """Generate all legal moves for the current position."""
        if not self.board:
            return []

        moves = []
        color = self.board.turn

        for coord, piece in self.board.pieces_of_color(color):
            piece_moves = self._generate_piece_moves(coord, piece)
            moves.extend(piece_moves)

        return moves

    def _generate_piece_moves(self, coord: Coord, piece: Piece) -> List[Move]:
        """Generate moves for a single piece (simplified)."""
        moves = []
        file, rank = coord

        # Simplified move generation (not handling all rules)
        if piece.piece_type == PieceType.PAWN:
            direction = 1 if piece.color == Color.WHITE else -1
            # Forward move
            target = (file, rank + direction)
            if self.board.is_in_bounds(target) and not self.board.get_piece(target):
                if (piece.color == Color.WHITE and rank + direction == 7) or \
                   (piece.color == Color.BLACK and rank + direction == 0):
                    # Promotion
                    for promo in [PieceType.QUEEN, PieceType.ROOK, PieceType.BISHOP, PieceType.KNIGHT]:
                        moves.append(Move(coord, target, promotion=promo))
                else:
                    moves.append(Move(coord, target))

            # Captures (diagonal)
            for df in [-1, 1]:
                target = (file + df, rank + direction)
                if self.board.is_in_bounds(target):
                    target_piece = self.board.get_piece(target)
                    if target_piece and target_piece.color != piece.color:
                        moves.append(Move(coord, target))

        elif piece.piece_type == PieceType.KNIGHT:
            for df, dr in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
                target = (file + df, rank + dr)
                if self.board.is_in_bounds(target):
                    target_piece = self.board.get_piece(target)
                    if not target_piece or target_piece.color != piece.color:
                        moves.append(Move(coord, target))

        elif piece.piece_type == PieceType.KING:
            for df in [-1, 0, 1]:
                for dr in [-1, 0, 1]:
                    if df == 0 and dr == 0:
                        continue
                    target = (file + df, rank + dr)
                    if self.board.is_in_bounds(target):
                        target_piece = self.board.get_piece(target)
                        if not target_piece or target_piece.color != piece.color:
                            moves.append(Move(coord, target))

        elif piece.piece_type in [PieceType.BISHOP, PieceType.ROOK, PieceType.QUEEN]:
            directions = []
            if piece.piece_type in [PieceType.BISHOP, PieceType.QUEEN]:
                directions.extend([(-1,-1), (-1,1), (1,-1), (1,1)])
            if piece.piece_type in [PieceType.ROOK, PieceType.QUEEN]:
                directions.extend([(-1,0), (1,0), (0,-1), (0,1)])

            for df, dr in directions:
                for dist in range(1, 20):  # Extended for infinite boards
                    target = (file + df * dist, rank + dr * dist)
                    if not self.board.is_in_bounds(target):
                        break
                    target_piece = self.board.get_piece(target)
                    if not target_piece:
                        moves.append(Move(coord, target))
                    elif target_piece.color != piece.color:
                        moves.append(Move(coord, target))
                        break
                    else:
                        break

        return moves

    def evaluate(self) -> EngineEvaluation:
        """
        Evaluate the current position.

        Returns complete evaluation including ordinal game value.
        """
        start_time = time.time()

        if not self.board:
            raise ValueError("No position set")

        legal_moves = self._generate_legal_moves()

        if not legal_moves:
            # No legal moves = checkmate or stalemate
            # Simplified: assume stalemate (should check for check)
            return EngineEvaluation(
                best_move=None,
                ordinal_value=OrdinalValue.draw(),
                win_probability=0.0,
                draw_probability=1.0,
                loss_probability=0.0,
                top_moves=[],
                think_time_ms=int((time.time() - start_time) * 1000),
            )

        if self.model is None:
            # Random fallback
            import random
            best_move = random.choice(legal_moves)
            return EngineEvaluation(
                best_move=best_move,
                ordinal_value=OrdinalValue.from_ply(10),
                win_probability=0.5,
                draw_probability=0.3,
                loss_probability=0.2,
                top_moves=[(best_move, 0.0)],
                think_time_ms=int((time.time() - start_time) * 1000),
            )

        # Encode position
        perspective = self.board.turn
        encoding = self.encoder.encode_position(self.board, perspective)

        # Prepare inputs
        tokens = torch.tensor(encoding['tokens']).unsqueeze(0).to(self.device)
        rel_pos = torch.tensor(encoding['position_ids']).unsqueeze(0).to(self.device)
        turn = torch.tensor([perspective]).to(self.device)

        # Model inference
        with torch.no_grad():
            outputs = self.model(tokens, rel_pos, turn)

        # Extract predictions
        value_probs = outputs['value'][0].cpu().numpy()
        loss_prob, draw_prob, win_prob = value_probs

        ordinal_bucket = outputs['ordinal_predicted_bucket'][0].item()
        ordinal_value = self.model.bucketizer.from_bucket(ordinal_bucket)

        # Get action values for legal moves
        action_values = outputs['action_values'][0].cpu().numpy()

        # Map moves to indices and get scores
        move_scores = []
        for move in legal_moves:
            # Simplified move indexing
            move_idx = self._move_to_index(move) % len(action_values)
            score = action_values[move_idx]
            move_scores.append((move, score))

        # Sort by score
        move_scores.sort(key=lambda x: x[1], reverse=True)

        # Select best move with temperature
        if self.config.temperature > 0:
            scores = np.array([s for _, s in move_scores])
            probs = np.exp(scores / self.config.temperature)
            probs = probs / probs.sum()
            idx = np.random.choice(len(move_scores), p=probs)
            best_move = move_scores[idx][0]
        else:
            best_move = move_scores[0][0]

        elapsed_ms = int((time.time() - start_time) * 1000)

        return EngineEvaluation(
            best_move=best_move,
            ordinal_value=ordinal_value,
            win_probability=float(win_prob),
            draw_probability=float(draw_prob),
            loss_probability=float(loss_prob),
            top_moves=move_scores[:5],
            think_time_ms=elapsed_ms,
        )

    def _move_to_index(self, move: Move) -> int:
        """Convert move to action index (simplified)."""
        from_idx = move.from_square[0] + move.from_square[1] * 8
        to_idx = move.to_square[0] + move.to_square[1] * 8
        return from_idx * 64 + to_idx

    def go(
        self,
        wtime: int = None,
        btime: int = None,
        winc: int = 0,
        binc: int = 0,
        movetime: int = None,
    ) -> str:
        """
        Search for best move (UCI go command).

        Returns UCI bestmove string.
        """
        evaluation = self.evaluate()

        if evaluation.best_move:
            return f"bestmove {evaluation.best_move}"
        else:
            return "bestmove 0000"  # No legal moves

    def uci_loop(self):
        """Run UCI protocol loop."""
        print("OrdinalChess Engine v1.0")
        print("Based on searchless_chess + Evans-Hamkins transfinite game theory")

        while True:
            try:
                line = input().strip()
            except EOFError:
                break

            if not line:
                continue

            tokens = line.split()
            cmd = tokens[0]

            if cmd == "uci":
                print("id name OrdinalChess")
                print("id author OrdinalChess Team")
                print("option name ModelSize type combo default small var small var medium var large")
                print("uciok")

            elif cmd == "isready":
                print("readyok")

            elif cmd == "ucinewgame":
                self.new_game()

            elif cmd == "position":
                if len(tokens) > 1:
                    if tokens[1] == "startpos":
                        moves = tokens[3:] if len(tokens) > 3 and tokens[2] == "moves" else []
                        self.set_position("startpos", moves)
                    elif tokens[1] == "fen":
                        fen_end = tokens.index("moves") if "moves" in tokens else len(tokens)
                        fen = " ".join(tokens[2:fen_end])
                        moves = tokens[fen_end + 1:] if fen_end < len(tokens) else []
                        self.set_position(fen, moves)

            elif cmd == "go":
                # Parse go parameters
                params = {}
                i = 1
                while i < len(tokens):
                    if tokens[i] in ["wtime", "btime", "winc", "binc", "movetime"]:
                        params[tokens[i]] = int(tokens[i + 1])
                        i += 2
                    else:
                        i += 1

                evaluation = self.evaluate()
                print(evaluation.to_uci_info())
                print(self.go(**params))

            elif cmd == "eval":
                # Custom command: detailed evaluation
                evaluation = self.evaluate()
                print(f"Position evaluation:")
                print(f"  Ordinal value: {evaluation.ordinal_value}")
                print(f"  Ordinal tier: {evaluation.ordinal_value.tier().name}")
                print(f"  Win/Draw/Loss: {evaluation.win_probability:.3f}/{evaluation.draw_probability:.3f}/{evaluation.loss_probability:.3f}")
                print(f"  Best move: {evaluation.best_move}")
                print(f"  Top moves:")
                for move, score in evaluation.top_moves:
                    print(f"    {move}: {score:.3f}")

            elif cmd == "quit":
                break

            elif cmd == "d":
                # Display board
                if self.board:
                    print(self.board)


def main():
    """Run the engine in UCI mode."""
    engine = OrdinalChessEngine()
    engine.uci_loop()


if __name__ == "__main__":
    main()
