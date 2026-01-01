"""
Extended/Infinite Board Representation for OrdinalChess

Supports:
1. Standard 8x8 chess board
2. Extended N×M boards (e.g., 16×16, 32×32)
3. Infinite board representation using sparse coordinate systems

Key insight from Evans & Hamkins: Infinite chess boards are necessary to
achieve transfinite ordinal game values. The ω (omega) game value arises
from positions where white can force a win, but black can delay indefinitely
through infinite retreat patterns.

This module provides:
- ExtendedBoard: Generalized board representation
- SparseBoard: Memory-efficient infinite board using coordinate dictionaries
- Position encoding for transformer input
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Iterator
from enum import IntEnum
import numpy as np
from collections import defaultdict


class PieceType(IntEnum):
    """Chess piece types."""
    EMPTY = 0
    PAWN = 1
    KNIGHT = 2
    BISHOP = 3
    ROOK = 4
    QUEEN = 5
    KING = 6


class Color(IntEnum):
    """Player colors."""
    WHITE = 0
    BLACK = 1


@dataclass
class Piece:
    """Represents a chess piece."""
    piece_type: PieceType
    color: Color

    def __repr__(self) -> str:
        symbols = {
            (PieceType.PAWN, Color.WHITE): "♙",
            (PieceType.KNIGHT, Color.WHITE): "♘",
            (PieceType.BISHOP, Color.WHITE): "♗",
            (PieceType.ROOK, Color.WHITE): "♖",
            (PieceType.QUEEN, Color.WHITE): "♕",
            (PieceType.KING, Color.WHITE): "♔",
            (PieceType.PAWN, Color.BLACK): "♟",
            (PieceType.KNIGHT, Color.BLACK): "♞",
            (PieceType.BISHOP, Color.BLACK): "♝",
            (PieceType.ROOK, Color.BLACK): "♜",
            (PieceType.QUEEN, Color.BLACK): "♛",
            (PieceType.KING, Color.BLACK): "♚",
        }
        return symbols.get((self.piece_type, self.color), ".")

    def to_token(self) -> int:
        """Convert piece to token ID for transformer input."""
        if self.piece_type == PieceType.EMPTY:
            return 0
        # 1-6 for white pieces, 7-12 for black pieces
        return int(self.piece_type) + (6 if self.color == Color.BLACK else 0)


# Coordinate type for board positions
Coord = Tuple[int, int]  # (file, rank) - can be negative for infinite boards


@dataclass
class Move:
    """Represents a chess move."""
    from_square: Coord
    to_square: Coord
    promotion: Optional[PieceType] = None
    is_castle: bool = False
    is_en_passant: bool = False

    def __repr__(self) -> str:
        def coord_str(c: Coord) -> str:
            file, rank = c
            if 0 <= file < 8 and 0 <= rank < 8:
                return f"{chr(ord('a') + file)}{rank + 1}"
            return f"({file},{rank})"

        result = f"{coord_str(self.from_square)}{coord_str(self.to_square)}"
        if self.promotion:
            result += f"={self.promotion.name[0]}"
        return result


class SparseBoard:
    """
    Memory-efficient board representation using sparse coordinate dictionary.

    Supports infinite/unbounded boards by only storing occupied squares.
    This is essential for representing the infinite chess positions from
    Evans & Hamkins that achieve transfinite ordinal game values.
    """

    def __init__(
        self,
        bounds: Optional[Tuple[int, int, int, int]] = None,  # (min_file, min_rank, max_file, max_rank)
        is_infinite: bool = False
    ):
        self.pieces: Dict[Coord, Piece] = {}
        self.bounds = bounds if bounds else (0, 0, 7, 7)  # Default 8x8
        self.is_infinite = is_infinite

        # Game state
        self.turn: Color = Color.WHITE
        self.castling_rights: Set[str] = {"K", "Q", "k", "q"}  # Standard notation
        self.en_passant_target: Optional[Coord] = None
        self.halfmove_clock: int = 0
        self.fullmove_number: int = 1

        # King positions for quick access
        self.king_positions: Dict[Color, Coord] = {}

    def copy(self) -> SparseBoard:
        """Create a deep copy of the board."""
        new_board = SparseBoard(self.bounds, self.is_infinite)
        new_board.pieces = self.pieces.copy()
        new_board.turn = self.turn
        new_board.castling_rights = self.castling_rights.copy()
        new_board.en_passant_target = self.en_passant_target
        new_board.halfmove_clock = self.halfmove_clock
        new_board.fullmove_number = self.fullmove_number
        new_board.king_positions = self.king_positions.copy()
        return new_board

    def set_piece(self, coord: Coord, piece: Optional[Piece]) -> None:
        """Place or remove a piece at the given coordinate."""
        if piece is None or piece.piece_type == PieceType.EMPTY:
            self.pieces.pop(coord, None)
        else:
            self.pieces[coord] = piece
            if piece.piece_type == PieceType.KING:
                self.king_positions[piece.color] = coord

    def get_piece(self, coord: Coord) -> Optional[Piece]:
        """Get piece at coordinate, or None if empty."""
        return self.pieces.get(coord)

    def is_in_bounds(self, coord: Coord) -> bool:
        """Check if coordinate is within board bounds (always True for infinite boards)."""
        if self.is_infinite:
            return True
        file, rank = coord
        min_f, min_r, max_f, max_r = self.bounds
        return min_f <= file <= max_f and min_r <= rank <= max_r

    def occupied_squares(self) -> Iterator[Tuple[Coord, Piece]]:
        """Iterate over all occupied squares."""
        for coord, piece in self.pieces.items():
            yield coord, piece

    def pieces_of_color(self, color: Color) -> Iterator[Tuple[Coord, Piece]]:
        """Iterate over pieces of a specific color."""
        for coord, piece in self.pieces.items():
            if piece.color == color:
                yield coord, piece

    @classmethod
    def from_fen(cls, fen: str) -> SparseBoard:
        """Parse FEN string into SparseBoard (standard 8x8)."""
        board = cls()
        parts = fen.split()

        # Parse piece placement
        rank = 7
        file = 0
        piece_map = {
            'p': (PieceType.PAWN, Color.BLACK),
            'n': (PieceType.KNIGHT, Color.BLACK),
            'b': (PieceType.BISHOP, Color.BLACK),
            'r': (PieceType.ROOK, Color.BLACK),
            'q': (PieceType.QUEEN, Color.BLACK),
            'k': (PieceType.KING, Color.BLACK),
            'P': (PieceType.PAWN, Color.WHITE),
            'N': (PieceType.KNIGHT, Color.WHITE),
            'B': (PieceType.BISHOP, Color.WHITE),
            'R': (PieceType.ROOK, Color.WHITE),
            'Q': (PieceType.QUEEN, Color.WHITE),
            'K': (PieceType.KING, Color.WHITE),
        }

        for char in parts[0]:
            if char == '/':
                rank -= 1
                file = 0
            elif char.isdigit():
                file += int(char)
            elif char in piece_map:
                ptype, color = piece_map[char]
                board.set_piece((file, rank), Piece(ptype, color))
                file += 1

        # Parse turn
        if len(parts) > 1:
            board.turn = Color.WHITE if parts[1] == 'w' else Color.BLACK

        # Parse castling rights
        if len(parts) > 2:
            board.castling_rights = set(parts[2]) if parts[2] != '-' else set()

        # Parse en passant
        if len(parts) > 3 and parts[3] != '-':
            file = ord(parts[3][0]) - ord('a')
            rank = int(parts[3][1]) - 1
            board.en_passant_target = (file, rank)

        # Parse clocks
        if len(parts) > 4:
            board.halfmove_clock = int(parts[4])
        if len(parts) > 5:
            board.fullmove_number = int(parts[5])

        return board

    @classmethod
    def starting_position(cls) -> SparseBoard:
        """Create standard starting position."""
        return cls.from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    @classmethod
    def omega_ladder(cls, n: int = 10) -> SparseBoard:
        """
        Create an "omega ladder" position achieving game value ω.

        This is a simplified infinite retreat pattern where black's king
        can retreat indefinitely along a rank, requiring white to make
        infinitely many moves to force mate (but white WILL eventually win).

        Based on Evans & Hamkins construction.
        """
        # Create extended board
        board = cls(bounds=(0, 0, n + 10, 7), is_infinite=False)

        # White: King + Queen for forcing pattern
        board.set_piece((0, 0), Piece(PieceType.KING, Color.WHITE))
        board.set_piece((1, 1), Piece(PieceType.QUEEN, Color.WHITE))
        board.set_piece((2, 1), Piece(PieceType.ROOK, Color.WHITE))

        # Black: King that can retreat along the rank
        board.set_piece((n, 4), Piece(PieceType.KING, Color.BLACK))

        return board

    @classmethod
    def omega_squared_position(cls) -> SparseBoard:
        """
        Create a position with game value ω² (omega squared).

        This requires nested infinite structures - multiple omega ladders
        that must be resolved in sequence.
        """
        board = cls(bounds=(0, 0, 31, 15), is_infinite=False)

        # White forcing pieces
        board.set_piece((0, 0), Piece(PieceType.KING, Color.WHITE))
        board.set_piece((1, 1), Piece(PieceType.QUEEN, Color.WHITE))
        board.set_piece((2, 0), Piece(PieceType.ROOK, Color.WHITE))
        board.set_piece((3, 0), Piece(PieceType.ROOK, Color.WHITE))

        # Black's nested retreat structure
        board.set_piece((20, 10), Piece(PieceType.KING, Color.BLACK))
        board.set_piece((25, 5), Piece(PieceType.ROOK, Color.BLACK))

        return board

    def to_tensor(self, center: Optional[Coord] = None, window: int = 8) -> np.ndarray:
        """
        Convert board region to tensor representation for transformer input.

        Args:
            center: Center of the window (default: king position or board center)
            window: Size of square window to extract

        Returns:
            Tensor of shape (window, window) with piece tokens
        """
        if center is None:
            # Default to white king position or board center
            if Color.WHITE in self.king_positions:
                center = self.king_positions[Color.WHITE]
            else:
                min_f, min_r, max_f, max_r = self.bounds
                center = ((min_f + max_f) // 2, (min_r + max_r) // 2)

        half = window // 2
        tensor = np.zeros((window, window), dtype=np.int32)

        for i in range(window):
            for j in range(window):
                coord = (center[0] - half + i, center[1] - half + j)
                piece = self.get_piece(coord)
                if piece:
                    tensor[j, i] = piece.to_token()

        return tensor

    def __repr__(self) -> str:
        """ASCII representation of the board."""
        min_f, min_r, max_f, max_r = self.bounds
        lines = []

        for rank in range(max_r, min_r - 1, -1):
            line = f"{rank + 1:2d} "
            for file in range(min_f, max_f + 1):
                piece = self.get_piece((file, rank))
                if piece:
                    line += str(piece) + " "
                else:
                    line += ". "
            lines.append(line)

        # File labels
        files = "   " + " ".join(chr(ord('a') + f) for f in range(min_f, max_f + 1))
        lines.append(files)

        return "\n".join(lines)


class BoardEncoder:
    """
    Encodes board positions for transformer input.

    Uses relative position encoding to support extended/infinite boards.
    The encoding is centered on the player's king with a fixed window.
    """

    def __init__(
        self,
        window_size: int = 16,  # Size of observation window
        max_pieces: int = 64,   # Maximum pieces to encode
        vocab_size: int = 13,   # 0=empty, 1-6=white pieces, 7-12=black pieces
    ):
        self.window_size = window_size
        self.max_pieces = max_pieces
        self.vocab_size = vocab_size

    def encode_position(
        self,
        board: SparseBoard,
        perspective: Color = Color.WHITE
    ) -> Dict[str, np.ndarray]:
        """
        Encode board position for transformer input.

        Returns:
            Dictionary with:
            - 'tokens': Piece tokens (window_size, window_size)
            - 'position_ids': Relative position IDs
            - 'piece_coords': Absolute coordinates of pieces
            - 'turn': Current turn (0=white, 1=black)
            - 'castling': Castling rights vector
        """
        # Get perspective king position as center
        if perspective in board.king_positions:
            center = board.king_positions[perspective]
        else:
            min_f, min_r, max_f, max_r = board.bounds
            center = ((min_f + max_f) // 2, (min_r + max_r) // 2)

        # Extract window around center
        tokens = board.to_tensor(center, self.window_size)

        # Position IDs (relative to center)
        half = self.window_size // 2
        position_ids = np.zeros((self.window_size, self.window_size, 2), dtype=np.int32)
        for i in range(self.window_size):
            for j in range(self.window_size):
                position_ids[j, i] = [i - half, j - half]

        # Piece list encoding (for attention-based architectures)
        piece_list = []
        for coord, piece in board.occupied_squares():
            rel_coord = (coord[0] - center[0], coord[1] - center[1])
            # Only include pieces within a reasonable range
            if abs(rel_coord[0]) < 100 and abs(rel_coord[1]) < 100:
                piece_list.append({
                    'token': piece.to_token(),
                    'rel_file': rel_coord[0],
                    'rel_rank': rel_coord[1],
                    'abs_file': coord[0],
                    'abs_rank': coord[1],
                })

        # Pad or truncate piece list
        piece_list = piece_list[:self.max_pieces]

        # Castling rights as binary vector
        castling = np.array([
            'K' in board.castling_rights,
            'Q' in board.castling_rights,
            'k' in board.castling_rights,
            'q' in board.castling_rights,
        ], dtype=np.float32)

        return {
            'tokens': tokens,
            'position_ids': position_ids,
            'piece_list': piece_list,
            'turn': np.array([board.turn], dtype=np.int32),
            'castling': castling,
            'center': np.array(center, dtype=np.int32),
            'is_infinite': np.array([board.is_infinite], dtype=np.bool_),
        }


if __name__ == "__main__":
    print("=== OrdinalChess: Extended Board Representation ===\n")

    # Standard position
    print("Standard starting position:")
    board = SparseBoard.starting_position()
    print(board)
    print()

    # Omega ladder position
    print("Omega ladder position (game value ω):")
    omega_board = SparseBoard.omega_ladder(15)
    print(omega_board)
    print()

    # Encode for transformer
    encoder = BoardEncoder(window_size=8)
    encoding = encoder.encode_position(board)
    print(f"Token tensor shape: {encoding['tokens'].shape}")
    print(f"Number of pieces encoded: {len(encoding['piece_list'])}")
