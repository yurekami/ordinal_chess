"""
Infinite-Dimensional Chess Board Representation

Based on Bolan & Tsevas (2024) "Transfinite game values in infinite-dimensional chess"
https://tsevasa.github.io/infinite_dimensional_chess/

Key Mathematical Concepts:
1. Board is an infinite-dimensional integer lattice
2. Distance measured by supremum norm: ||x||∞ = sup{|x₁|, |x₂|, ...}
3. Kings move within ||·||∞ distance 1 (can change multiple coords by ±1)
4. Rooks move along single coordinate axis
5. Game values can reach continuum cardinality (2^ℵ₀)

Variants:
- 8×8×8×... : coordinates in [-4, 3]
- 3×3×3×... : coordinates in [-1, 1]
- Weak pieces: only finitely many coordinates change per move
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import (
    Optional, List, Dict, Tuple, Set, Iterator, Callable,
    Union, FrozenSet, Sequence
)
from enum import IntEnum
from collections import defaultdict
import numpy as np
from functools import cached_property
import itertools

from .board import PieceType, Color, Piece


# Type for infinite-dimensional coordinates
# Represented as a dictionary mapping dimension index to value
# Missing indices are assumed to be 0 (sparse representation)
InfCoord = Dict[int, int]


@dataclass(frozen=True)
class Coordinate:
    """
    Represents a coordinate in N-dimensional or infinite-dimensional space.

    Uses sparse representation: only non-zero coordinates are stored.
    For infinite dimensions, coordinates beyond the stored ones are assumed 0.

    Properties:
    - Immutable (frozen dataclass)
    - Hashable (can be used as dict keys)
    - Supports supremum norm distance
    """
    # Sparse storage: dimension_index -> value
    _values: Tuple[Tuple[int, int], ...]  # Sorted tuple of (dim, value) pairs

    @classmethod
    def from_dict(cls, d: Dict[int, int]) -> Coordinate:
        """Create coordinate from dictionary."""
        # Filter out zeros and sort by dimension
        values = tuple(sorted((k, v) for k, v in d.items() if v != 0))
        return cls(_values=values)

    @classmethod
    def from_list(cls, lst: Sequence[int]) -> Coordinate:
        """Create coordinate from list [x₀, x₁, x₂, ...]."""
        d = {i: v for i, v in enumerate(lst) if v != 0}
        return cls.from_dict(d)

    @classmethod
    def zero(cls) -> Coordinate:
        """Create the zero coordinate (origin)."""
        return cls(_values=())

    @classmethod
    def unit(cls, dim: int, value: int = 1) -> Coordinate:
        """Create unit vector in given dimension."""
        if value == 0:
            return cls.zero()
        return cls(_values=((dim, value),))

    def __getitem__(self, dim: int) -> int:
        """Get coordinate value in dimension dim."""
        for d, v in self._values:
            if d == dim:
                return v
            if d > dim:
                break
        return 0

    def to_dict(self) -> Dict[int, int]:
        """Convert to dictionary representation."""
        return dict(self._values)

    def to_list(self, n_dims: int) -> List[int]:
        """Convert to list representation for first n dimensions."""
        result = [0] * n_dims
        for d, v in self._values:
            if d < n_dims:
                result[d] = v
        return result

    @cached_property
    def dimensions_used(self) -> FrozenSet[int]:
        """Set of dimensions with non-zero values."""
        return frozenset(d for d, v in self._values)

    @cached_property
    def max_dimension(self) -> int:
        """Highest dimension index with non-zero value (-1 if zero coord)."""
        if not self._values:
            return -1
        return self._values[-1][0]

    @cached_property
    def sup_norm(self) -> int:
        """Supremum norm: ||x||∞ = sup{|xᵢ|}."""
        if not self._values:
            return 0
        return max(abs(v) for _, v in self._values)

    def distance(self, other: Coordinate) -> int:
        """Compute ||self - other||∞ (supremum norm distance)."""
        return (self - other).sup_norm

    def __add__(self, other: Coordinate) -> Coordinate:
        """Vector addition."""
        result = dict(self._values)
        for d, v in other._values:
            result[d] = result.get(d, 0) + v
        return Coordinate.from_dict(result)

    def __sub__(self, other: Coordinate) -> Coordinate:
        """Vector subtraction."""
        result = dict(self._values)
        for d, v in other._values:
            result[d] = result.get(d, 0) - v
        return Coordinate.from_dict(result)

    def __neg__(self) -> Coordinate:
        """Negation."""
        return Coordinate._values(tuple((d, -v) for d, v in self._values))

    def __mul__(self, scalar: int) -> Coordinate:
        """Scalar multiplication."""
        if scalar == 0:
            return Coordinate.zero()
        return Coordinate(tuple((d, v * scalar) for d, v in self._values))

    def __repr__(self) -> str:
        if not self._values:
            return "0⃗"
        parts = [f"x{d}={v}" for d, v in self._values[:5]]
        if len(self._values) > 5:
            parts.append("...")
        return f"({', '.join(parts)})"

    def __hash__(self) -> int:
        return hash(self._values)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Coordinate):
            return False
        return self._values == other._values


class BoardVariant(IntEnum):
    """Predefined board variants from Bolan-Tsevas."""
    STANDARD_2D = 0      # Traditional 8×8
    EXTENDED_3D = 1      # Evans-Hamkins style
    INFINITE_8 = 2       # 8×8×8×... (coords in [-4, 3])
    INFINITE_3 = 3       # 3×3×3×... (coords in [-1, 1])
    CUSTOM = 4           # User-defined


@dataclass
class BoardConfig:
    """Configuration for N-dimensional/infinite board."""
    variant: BoardVariant = BoardVariant.INFINITE_8

    # Coordinate bounds (per dimension)
    min_coord: int = -4  # Minimum coordinate value
    max_coord: int = 3   # Maximum coordinate value

    # Dimension limits
    n_dimensions: Optional[int] = None  # None = infinite

    # Piece rules
    weak_pieces: bool = False  # If True, only finitely many coords can change
    max_coord_changes: int = 8  # For weak pieces: max coords changed per move

    # Principal axis for pawns (dimension 0 by default)
    principal_axis: int = 0

    @classmethod
    def infinite_8(cls) -> BoardConfig:
        """8×8×8×... variant."""
        return cls(
            variant=BoardVariant.INFINITE_8,
            min_coord=-4,
            max_coord=3,
            n_dimensions=None,
        )

    @classmethod
    def infinite_3(cls) -> BoardConfig:
        """3×3×3×... variant."""
        return cls(
            variant=BoardVariant.INFINITE_3,
            min_coord=-1,
            max_coord=1,
            n_dimensions=None,
        )

    @classmethod
    def weak_pieces(cls) -> BoardConfig:
        """Weak pieces variant (countable ordinals only)."""
        return cls(
            variant=BoardVariant.INFINITE_8,
            min_coord=-4,
            max_coord=3,
            n_dimensions=None,
            weak_pieces=True,
            max_coord_changes=8,
        )

    @property
    def sidelength(self) -> int:
        """Board sidelength (max_coord - min_coord + 1)."""
        return self.max_coord - self.min_coord + 1

    def is_valid_coord_value(self, value: int) -> bool:
        """Check if coordinate value is within bounds."""
        return self.min_coord <= value <= self.max_coord

    def is_valid_coordinate(self, coord: Coordinate) -> bool:
        """Check if all coordinate values are within bounds."""
        for _, v in coord._values:
            if not self.is_valid_coord_value(v):
                return False
        return True


@dataclass
class InfDimMove:
    """Represents a move in infinite-dimensional chess."""
    from_coord: Coordinate
    to_coord: Coordinate
    piece_type: PieceType
    is_capture: bool = False
    captured_piece: Optional[PieceType] = None

    @cached_property
    def coords_changed(self) -> FrozenSet[int]:
        """Set of dimensions that changed."""
        changed = set()
        from_dict = self.from_coord.to_dict()
        to_dict = self.to_coord.to_dict()
        all_dims = set(from_dict.keys()) | set(to_dict.keys())
        for d in all_dims:
            if from_dict.get(d, 0) != to_dict.get(d, 0):
                changed.add(d)
        return frozenset(changed)

    @cached_property
    def n_coords_changed(self) -> int:
        """Number of coordinates that changed."""
        return len(self.coords_changed)

    @cached_property
    def displacement(self) -> Coordinate:
        """Movement vector."""
        return self.to_coord - self.from_coord

    @cached_property
    def distance(self) -> int:
        """Supremum norm distance of the move."""
        return self.displacement.sup_norm

    def is_valid_king_move(self) -> bool:
        """Check if this is a valid king move (distance ≤ 1)."""
        return self.distance <= 1

    def is_valid_rook_move(self) -> bool:
        """Check if this is a valid rook move (single axis)."""
        return self.n_coords_changed == 1

    def is_valid_weak_move(self, max_changes: int) -> bool:
        """Check if move satisfies weak pieces constraint."""
        return self.n_coords_changed <= max_changes

    def __repr__(self) -> str:
        cap = "x" if self.is_capture else "-"
        return f"{self.piece_type.name}{self.from_coord}{cap}{self.to_coord}"


class InfiniteDimensionalBoard:
    """
    Board representation for infinite-dimensional chess.

    Uses sparse storage for pieces on an infinite-dimensional lattice.
    Supports the Bolan-Tsevas construction for achieving continuum ordinals.
    """

    def __init__(self, config: Optional[BoardConfig] = None):
        self.config = config or BoardConfig.infinite_8()

        # Sparse piece storage: Coordinate -> Piece
        self.pieces: Dict[Coordinate, Piece] = {}

        # Game state
        self.turn: Color = Color.WHITE
        self.move_history: List[InfDimMove] = []

        # King positions for quick access
        self.king_positions: Dict[Color, Coordinate] = {}

        # Cache for piece locations by color
        self._pieces_by_color: Dict[Color, Set[Coordinate]] = {
            Color.WHITE: set(),
            Color.BLACK: set(),
        }

    def copy(self) -> InfiniteDimensionalBoard:
        """Create a deep copy."""
        new_board = InfiniteDimensionalBoard(self.config)
        new_board.pieces = self.pieces.copy()
        new_board.turn = self.turn
        new_board.move_history = self.move_history.copy()
        new_board.king_positions = self.king_positions.copy()
        new_board._pieces_by_color = {
            c: s.copy() for c, s in self._pieces_by_color.items()
        }
        return new_board

    def set_piece(self, coord: Coordinate, piece: Optional[Piece]) -> None:
        """Place or remove a piece."""
        old_piece = self.pieces.get(coord)

        # Remove old piece from caches
        if old_piece:
            self._pieces_by_color[old_piece.color].discard(coord)
            if old_piece.piece_type == PieceType.KING:
                del self.king_positions[old_piece.color]

        # Set new piece
        if piece and piece.piece_type != PieceType.EMPTY:
            self.pieces[coord] = piece
            self._pieces_by_color[piece.color].add(coord)
            if piece.piece_type == PieceType.KING:
                self.king_positions[piece.color] = coord
        else:
            self.pieces.pop(coord, None)

    def get_piece(self, coord: Coordinate) -> Optional[Piece]:
        """Get piece at coordinate."""
        return self.pieces.get(coord)

    def is_occupied(self, coord: Coordinate) -> bool:
        """Check if coordinate has a piece."""
        return coord in self.pieces

    def pieces_of_color(self, color: Color) -> Iterator[Tuple[Coordinate, Piece]]:
        """Iterate over pieces of a color."""
        for coord in self._pieces_by_color[color]:
            yield coord, self.pieces[coord]

    def generate_king_moves(self, from_coord: Coordinate) -> Iterator[Coordinate]:
        """
        Generate all valid king moves from a position.

        King can move to any square within ||·||∞ distance 1.
        This means changing any subset of coordinates by ±1.
        """
        piece = self.get_piece(from_coord)
        if not piece or piece.piece_type != PieceType.KING:
            return

        # Find all non-zero dimensions in current position
        active_dims = set(from_coord.dimensions_used)

        # Also consider adjacent dimensions (for expansion)
        if from_coord.max_dimension >= 0:
            active_dims.add(from_coord.max_dimension + 1)
        else:
            active_dims.add(0)

        # For weak pieces, limit dimensions considered
        if self.config.weak_pieces:
            active_dims = set(list(active_dims)[:self.config.max_coord_changes])

        # Generate all combinations of ±1 changes
        # For efficiency, limit to reasonable number of dimensions
        dims_list = sorted(active_dims)[:12]  # Practical limit

        for r in range(len(dims_list) + 1):
            for dim_subset in itertools.combinations(dims_list, r):
                if r == 0:
                    continue  # Skip staying in place

                # Generate all sign combinations
                for signs in itertools.product([-1, 1], repeat=r):
                    delta = {}
                    valid = True
                    for dim, sign in zip(dim_subset, signs):
                        new_val = from_coord[dim] + sign
                        if not self.config.is_valid_coord_value(new_val):
                            valid = False
                            break
                        if new_val != 0:
                            delta[dim] = new_val
                        # If new_val == 0, don't include (sparse)

                    if valid:
                        # Create target coordinate
                        new_coord_dict = from_coord.to_dict()
                        for dim in dim_subset:
                            new_coord_dict.pop(dim, None)  # Remove old
                        new_coord_dict.update(delta)
                        target = Coordinate.from_dict(new_coord_dict)

                        # Check not occupied by own piece
                        target_piece = self.get_piece(target)
                        if not target_piece or target_piece.color != piece.color:
                            yield target

    def generate_rook_moves(self, from_coord: Coordinate) -> Iterator[Coordinate]:
        """
        Generate all valid rook moves from a position.

        Rook moves along a single coordinate axis (changes one coordinate).
        """
        piece = self.get_piece(from_coord)
        if not piece or piece.piece_type != PieceType.ROOK:
            return

        # Consider all dimensions with non-zero values plus dimension 0
        dims_to_try = set(from_coord.dimensions_used)
        dims_to_try.add(0)
        if from_coord.max_dimension >= 0:
            dims_to_try.add(from_coord.max_dimension + 1)

        for dim in dims_to_try:
            current_val = from_coord[dim]

            # Move in positive direction
            for delta in range(1, self.config.sidelength + 1):
                new_val = current_val + delta
                if not self.config.is_valid_coord_value(new_val):
                    break

                new_dict = from_coord.to_dict()
                if new_val == 0:
                    new_dict.pop(dim, None)
                else:
                    new_dict[dim] = new_val
                target = Coordinate.from_dict(new_dict)

                target_piece = self.get_piece(target)
                if target_piece:
                    if target_piece.color != piece.color:
                        yield target  # Capture
                    break  # Blocked
                yield target

            # Move in negative direction
            for delta in range(1, self.config.sidelength + 1):
                new_val = current_val - delta
                if not self.config.is_valid_coord_value(new_val):
                    break

                new_dict = from_coord.to_dict()
                if new_val == 0:
                    new_dict.pop(dim, None)
                else:
                    new_dict[dim] = new_val
                target = Coordinate.from_dict(new_dict)

                target_piece = self.get_piece(target)
                if target_piece:
                    if target_piece.color != piece.color:
                        yield target
                    break
                yield target

    def generate_pawn_moves(self, from_coord: Coordinate) -> Iterator[Coordinate]:
        """
        Generate pawn moves along principal axis.

        White pawns increase principal coordinate, black decrease.
        """
        piece = self.get_piece(from_coord)
        if not piece or piece.piece_type != PieceType.PAWN:
            return

        axis = self.config.principal_axis
        direction = 1 if piece.color == Color.WHITE else -1
        current_val = from_coord[axis]
        new_val = current_val + direction

        if not self.config.is_valid_coord_value(new_val):
            return

        # Forward move
        new_dict = from_coord.to_dict()
        if new_val == 0:
            new_dict.pop(axis, None)
        else:
            new_dict[axis] = new_val
        target = Coordinate.from_dict(new_dict)

        if not self.is_occupied(target):
            yield target

    def generate_moves(self, from_coord: Coordinate) -> Iterator[InfDimMove]:
        """Generate all legal moves for piece at coordinate."""
        piece = self.get_piece(from_coord)
        if not piece:
            return

        if piece.piece_type == PieceType.KING:
            targets = self.generate_king_moves(from_coord)
        elif piece.piece_type == PieceType.ROOK:
            targets = self.generate_rook_moves(from_coord)
        elif piece.piece_type == PieceType.PAWN:
            targets = self.generate_pawn_moves(from_coord)
        else:
            return  # Other pieces not implemented for infinite chess

        for target in targets:
            target_piece = self.get_piece(target)
            move = InfDimMove(
                from_coord=from_coord,
                to_coord=target,
                piece_type=piece.piece_type,
                is_capture=target_piece is not None,
                captured_piece=target_piece.piece_type if target_piece else None,
            )

            # Check weak pieces constraint
            if self.config.weak_pieces:
                if not move.is_valid_weak_move(self.config.max_coord_changes):
                    continue

            yield move

    def apply_move(self, move: InfDimMove) -> None:
        """Apply a move to the board."""
        piece = self.get_piece(move.from_coord)
        if not piece:
            raise ValueError(f"No piece at {move.from_coord}")

        self.set_piece(move.from_coord, None)
        self.set_piece(move.to_coord, piece)

        self.move_history.append(move)
        self.turn = Color.BLACK if self.turn == Color.WHITE else Color.WHITE

    def is_in_check(self, color: Color) -> bool:
        """Check if the given color's king is in check."""
        if color not in self.king_positions:
            return False

        king_coord = self.king_positions[color]
        opponent = Color.BLACK if color == Color.WHITE else Color.WHITE

        # Check if any opponent piece can capture the king
        for coord, piece in self.pieces_of_color(opponent):
            for move in self.generate_moves(coord):
                if move.to_coord == king_coord:
                    return True
        return False

    def is_checkmate(self, color: Color) -> bool:
        """Check if the given color is in checkmate."""
        if not self.is_in_check(color):
            return False

        # Try all possible moves to escape
        for coord, piece in list(self.pieces_of_color(color)):
            for move in self.generate_moves(coord):
                # Try the move
                test_board = self.copy()
                test_board.apply_move(move)
                if not test_board.is_in_check(color):
                    return False
        return True

    def display_slice(
        self,
        dims: Tuple[int, int] = (0, 1),
        fixed: Optional[Dict[int, int]] = None,
        size: int = 8
    ) -> str:
        """Display a 2D slice of the board."""
        fixed = fixed or {}
        d1, d2 = dims
        half = size // 2

        lines = []
        for y in range(half - 1, -half - 1, -1):
            row = f"{y:3d} "
            for x in range(-half, half):
                coord_dict = {d1: x, d2: y}
                coord_dict.update(fixed)
                coord = Coordinate.from_dict(coord_dict)
                piece = self.get_piece(coord)
                if piece:
                    symbols = {
                        (PieceType.KING, Color.WHITE): "♔",
                        (PieceType.KING, Color.BLACK): "♚",
                        (PieceType.ROOK, Color.WHITE): "♖",
                        (PieceType.ROOK, Color.BLACK): "♜",
                        (PieceType.PAWN, Color.WHITE): "♙",
                        (PieceType.PAWN, Color.BLACK): "♟",
                    }
                    row += symbols.get((piece.piece_type, piece.color), "?") + " "
                else:
                    row += ". "
            lines.append(row)

        # Add x-axis labels
        labels = "    " + " ".join(f"{x:2d}"[-2:] for x in range(-half, half))
        lines.append(labels)

        return "\n".join(lines)


def create_omega_position(n: int = 10) -> InfiniteDimensionalBoard:
    """
    Create a position with game value ω (omega).

    Uses the Bolan-Tsevas construction: Black king can retreat
    indefinitely along increasing dimensions, but White eventually wins.
    """
    board = InfiniteDimensionalBoard(BoardConfig.infinite_8())

    # White: King at origin, Rooks ready to deliver mate
    board.set_piece(Coordinate.from_list([0, 0]), Piece(PieceType.KING, Color.WHITE))
    board.set_piece(Coordinate.from_list([1, -3]), Piece(PieceType.ROOK, Color.WHITE))
    board.set_piece(Coordinate.from_list([2, -3]), Piece(PieceType.ROOK, Color.WHITE))

    # Black: King at distance n, can retreat infinitely
    board.set_piece(Coordinate.from_list([0] * n + [1]), Piece(PieceType.KING, Color.BLACK))

    return board


def create_omega_squared_position() -> InfiniteDimensionalBoard:
    """
    Create a position with game value ω² (omega squared).

    Nested infinite structure requiring omega-many omega ladders.
    """
    board = InfiniteDimensionalBoard(BoardConfig.infinite_8())

    # White configuration
    board.set_piece(Coordinate.from_list([0, 0, 0]), Piece(PieceType.KING, Color.WHITE))
    board.set_piece(Coordinate.from_list([1, -3, 0]), Piece(PieceType.ROOK, Color.WHITE))
    board.set_piece(Coordinate.from_list([2, -3, 0]), Piece(PieceType.ROOK, Color.WHITE))
    board.set_piece(Coordinate.from_list([3, -3, 0]), Piece(PieceType.ROOK, Color.WHITE))

    # Black: King in nested structure
    board.set_piece(Coordinate.from_list([0, 0, 0, 0, 1, 0, 1]),
                   Piece(PieceType.KING, Color.BLACK))

    return board


if __name__ == "__main__":
    print("=== Infinite-Dimensional Chess Test ===\n")

    # Test coordinate operations
    print("1. Coordinate Operations:")
    c1 = Coordinate.from_list([1, 2, 3, 0, 0, 1])
    c2 = Coordinate.from_list([2, 2, 2, 0, 0, 0])
    print(f"   c1 = {c1}")
    print(f"   c2 = {c2}")
    print(f"   ||c1||∞ = {c1.sup_norm}")
    print(f"   d(c1, c2) = {c1.distance(c2)}")
    print(f"   c1 + c2 = {c1 + c2}")

    # Test board
    print("\n2. Omega Position (game value ω):")
    board = create_omega_position()
    print(f"   White king: {board.king_positions.get(Color.WHITE)}")
    print(f"   Black king: {board.king_positions.get(Color.BLACK)}")
    print(f"   Distance between kings: {board.king_positions[Color.WHITE].distance(board.king_positions[Color.BLACK])}")

    # Test move generation
    print("\n3. King Move Generation:")
    white_king = board.king_positions[Color.WHITE]
    moves = list(board.generate_moves(white_king))
    print(f"   White king has {len(moves)} possible moves")
    if moves:
        print(f"   Sample moves: {moves[:3]}")

    print("\n=== Tests Complete ===")
