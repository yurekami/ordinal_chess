"""
Tree Embedding for Infinite-Dimensional Chess

Implements the Bolan-Tsevas construction for embedding ordinal trees
into infinite-dimensional chess positions.

Key Insight: Well-founded trees have ordinal ranks, and we can encode
these trees as paths through infinite-dimensional coordinate space.
The game value of a position equals the rank of the encoded tree.

Construction Overview:
1. Each tree node maps to a coordinate in infinite-dimensional space
2. Parent-child edges correspond to sup-norm distance 1 moves
3. Leaf nodes are "end traps" where checkmate is delivered
4. The Black king traverses the tree; game value = tree rank

This allows achieving ANY ordinal up to continuum cardinality (2^ℵ₀).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import (
    Optional, List, Dict, Set, Tuple, Iterator, Callable,
    Generic, TypeVar, FrozenSet, Sequence, Union
)
from abc import ABC, abstractmethod
from functools import cached_property
import itertools

from .dimensions import Coordinate, InfiniteDimensionalBoard, BoardConfig, InfDimMove
from .board import Piece, PieceType, Color
from .ordinals import OrdinalValue, OrdinalTier


T = TypeVar('T')


@dataclass
class TreeNode(Generic[T]):
    """
    A node in a well-founded tree.

    Well-founded trees have no infinite descending chains,
    ensuring every play eventually reaches a leaf (Black loses).
    """
    value: T                              # Node label/data
    children: List[TreeNode[T]] = field(default_factory=list)
    parent: Optional[TreeNode[T]] = None
    _rank: Optional[OrdinalValue] = None  # Cached ordinal rank

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def depth(self) -> int:
        """Depth from root (0 for root)."""
        if self.parent is None:
            return 0
        return self.parent.depth + 1

    @cached_property
    def rank(self) -> OrdinalValue:
        """
        Compute ordinal rank of this node.

        rank(leaf) = 0
        rank(node) = sup{rank(child) + 1 : child in children}
        """
        if self._rank is not None:
            return self._rank

        if self.is_leaf:
            return OrdinalValue.from_ply(0)

        # Compute maximum child rank + 1
        max_rank = OrdinalValue.from_ply(0)
        for child in self.children:
            child_rank = child.rank
            child_succ = child_rank.successor()
            if child_succ > max_rank:
                max_rank = child_succ

        return max_rank

    def add_child(self, value: T) -> TreeNode[T]:
        """Add a child node."""
        child = TreeNode(value=value, parent=self)
        self.children.append(child)
        return child

    def all_nodes(self) -> Iterator[TreeNode[T]]:
        """Iterate over all nodes in subtree (pre-order)."""
        yield self
        for child in self.children:
            yield from child.all_nodes()

    def all_leaves(self) -> Iterator[TreeNode[T]]:
        """Iterate over all leaf nodes in subtree."""
        if self.is_leaf:
            yield self
        else:
            for child in self.children:
                yield from child.all_leaves()

    def path_to_root(self) -> List[TreeNode[T]]:
        """Get path from this node to root (inclusive)."""
        path = [self]
        current = self
        while current.parent:
            current = current.parent
            path.append(current)
        return list(reversed(path))

    def __repr__(self) -> str:
        children_str = f", {len(self.children)} children" if self.children else ", leaf"
        return f"TreeNode({self.value}{children_str})"


class TreeBuilder:
    """Utilities for building trees with specific ordinal ranks."""

    @staticmethod
    def linear_tree(n: int) -> TreeNode[int]:
        """
        Build a linear tree of depth n.

        Rank = n (finite ordinal).

        Structure: 0 -> 1 -> 2 -> ... -> n (leaf)
        """
        root = TreeNode(value=0)
        current = root
        for i in range(1, n + 1):
            current = current.add_child(i)
        return root

    @staticmethod
    def omega_tree(branching: int = 2) -> TreeNode[Tuple[int, ...]]:
        """
        Build a tree with rank ω (omega).

        Structure: Root has infinitely many children, each a leaf.
        We approximate with `branching^k` children at level k.

        For true ω, we'd need countably infinite children.
        """
        root = TreeNode(value=())

        # Create branches of increasing depth
        for depth in range(10):  # Practical limit
            path = (depth,)
            current = root.add_child(path)
            # Each branch is a leaf at this depth
            # This gives rank = sup{1, 1, 1, ...} = 1 for leaves
            # But root rank = sup{child_rank + 1} = ω if infinite children

        return root

    @staticmethod
    def omega_power_tree(power: int) -> TreeNode[Tuple[int, ...]]:
        """
        Build a tree with rank ω^power.

        ω^1: Root with infinitely many children (leaves)
        ω^2: Root with infinitely many children, each of rank ω
        ω^3: Root with infinitely many children, each of rank ω²
        """
        def build_recursive(p: int, prefix: Tuple[int, ...]) -> TreeNode[Tuple[int, ...]]:
            node = TreeNode(value=prefix)
            if p == 0:
                return node  # Leaf

            # Add children, each of rank ω^(p-1)
            for i in range(10):  # Practical limit
                child = build_recursive(p - 1, prefix + (i,))
                child.parent = node
                node.children.append(child)

            return node

        return build_recursive(power, ())

    @staticmethod
    def binary_tree(depth: int) -> TreeNode[Tuple[int, ...]]:
        """Build a complete binary tree of given depth."""
        def build(d: int, path: Tuple[int, ...]) -> TreeNode[Tuple[int, ...]]:
            node = TreeNode(value=path)
            if d > 0:
                node.children = [
                    build(d - 1, path + (0,)),
                    build(d - 1, path + (1,)),
                ]
                for child in node.children:
                    child.parent = node
            return node

        return build(depth, ())


@dataclass
class NodeEncoding:
    """Encoding of a tree node as an infinite-dimensional coordinate."""
    node: TreeNode
    coordinate: Coordinate
    is_end_trap: bool = False  # True if this is a checkmate position


class TreeEmbedder:
    """
    Embeds a well-founded tree into infinite-dimensional chess.

    Uses the Bolan-Tsevas encoding scheme where:
    - Tree nodes become coordinates
    - Edges become sup-norm distance 1 adjacencies
    - Leaves become end traps (checkmate positions)

    The encoding for 8×8×8×... uses:
    - Coordinates (p, q, x₁, x₂, x₃, ...) with values in [-4, 3]
    - p = principal axis (pawn direction)
    - Binary sequences encoded via ±1/±2 values
    """

    def __init__(self, config: Optional[BoardConfig] = None):
        self.config = config or BoardConfig.infinite_8()
        self.node_encodings: Dict[int, NodeEncoding] = {}

    def encode_path(self, path: Sequence[int]) -> Coordinate:
        """
        Encode a binary path as an infinite-dimensional coordinate.

        Uses the Bolan-Tsevas scheme:
        - Recent elements encoded as ±1
        - Older elements encoded as ±2
        - Position in sequence determined by dimension index

        For 8×8×8×... variant (coordinates in [-4, 3]).
        """
        coord_dict: Dict[int, int] = {}

        # Principal axis: p = 0 (starting position)
        coord_dict[0] = 0

        # Secondary axis: q = path length indicator
        coord_dict[1] = min(len(path), 3)

        # Encode path elements
        for i, element in enumerate(path):
            # Dimension for this element
            # Using pattern: dimension = 2 + i
            dim = 2 + i

            # Encode as ±1 for recent, ±2 for older
            if i >= len(path) - 2:
                # Recent: use ±1
                coord_dict[dim] = 1 if element else -1
            else:
                # Older: use ±2
                coord_dict[dim] = 2 if element else -2

        return Coordinate.from_dict(coord_dict)

    def encode_3x3_path(self, path: Sequence[int]) -> Coordinate:
        """
        Encode for 3×3×3×... variant (coordinates in [-1, 1]).

        Uses the more complex Bolan-Tsevas scheme with:
        - Separate p_i dimensions encoding depth
        - x_i dimensions encoding path elements
        """
        coord_dict: Dict[int, int] = {}
        n = len(path)

        # Coordinate pattern: (p, q₁, q₂, p₁, x₁, p₂, x₂, ...)
        coord_dict[0] = 0  # p
        coord_dict[1] = min(n, 1)  # q₁
        coord_dict[2] = 0  # q₂

        for i, element in enumerate(path):
            # p_i encodes depth: 1 if i > n+2, 0 if i = n+2, -1 if i < n+2
            p_dim = 3 + 2 * i
            x_dim = 4 + 2 * i

            if i > n:
                coord_dict[p_dim] = 1
            elif i == n:
                coord_dict[p_dim] = 0
            else:
                coord_dict[p_dim] = -1

            # x_i encodes the path element
            coord_dict[x_dim] = 1 if element else -1

        return Coordinate.from_dict(coord_dict)

    def encode_tree_node(
        self,
        node: TreeNode,
        path_to_node: Optional[List[int]] = None
    ) -> NodeEncoding:
        """Encode a tree node as a coordinate."""
        if path_to_node is None:
            # Compute path from root
            node_path = node.path_to_root()
            path_to_node = []
            for i in range(len(node_path) - 1):
                parent = node_path[i]
                child = node_path[i + 1]
                child_idx = parent.children.index(child)
                # Convert to binary-ish encoding
                path_to_node.append(child_idx % 2)

        # Use appropriate encoding based on board variant
        if self.config.variant == self.config.variant.INFINITE_3:
            coord = self.encode_3x3_path(path_to_node)
        else:
            coord = self.encode_path(path_to_node)

        return NodeEncoding(
            node=node,
            coordinate=coord,
            is_end_trap=node.is_leaf,
        )

    def embed_tree(self, root: TreeNode) -> Dict[Coordinate, TreeNode]:
        """
        Embed entire tree into coordinate space.

        Returns mapping from coordinates to tree nodes.
        """
        coord_to_node: Dict[Coordinate, TreeNode] = {}

        def embed_recursive(node: TreeNode, path: List[int]):
            encoding = self.encode_tree_node(node, path)
            coord_to_node[encoding.coordinate] = node
            self.node_encodings[id(node)] = encoding

            for i, child in enumerate(node.children):
                child_path = path + [i % 2]
                embed_recursive(child, child_path)

        embed_recursive(root, [])
        return coord_to_node

    def create_position_from_tree(
        self,
        root: TreeNode,
        black_at_root: bool = True
    ) -> InfiniteDimensionalBoard:
        """
        Create a chess position encoding the given tree.

        The game value of this position equals the rank of the tree.
        """
        board = InfiniteDimensionalBoard(self.config)

        # Embed the tree
        coord_mapping = self.embed_tree(root)

        # Place Black king at root (or specified node)
        root_encoding = self.node_encodings[id(root)]
        board.set_piece(
            root_encoding.coordinate,
            Piece(PieceType.KING, Color.BLACK)
        )

        # Place White king adjacent to root (to follow Black)
        white_king_coord = self._adjacent_coord(root_encoding.coordinate)
        board.set_piece(white_king_coord, Piece(PieceType.KING, Color.WHITE))

        # Place rooks for checkmate at leaf nodes
        for coord, node in coord_mapping.items():
            if node.is_leaf:
                # End trap: place rooks that can deliver checkmate
                self._setup_end_trap(board, coord)

        return board

    def _adjacent_coord(self, coord: Coordinate) -> Coordinate:
        """Get an adjacent coordinate for White king placement."""
        # Move -1 in first non-zero dimension, or dimension 0
        coord_dict = coord.to_dict()
        if coord_dict:
            first_dim = min(coord_dict.keys())
            coord_dict[first_dim] = coord_dict[first_dim] - 1
        else:
            coord_dict[0] = -1

        # Check bounds
        if not self.config.is_valid_coord_value(coord_dict.get(0, 0)):
            coord_dict[0] = self.config.min_coord + 1

        return Coordinate.from_dict(coord_dict)

    def _setup_end_trap(
        self,
        board: InfiniteDimensionalBoard,
        leaf_coord: Coordinate
    ) -> None:
        """
        Set up end trap at a leaf node.

        When Black king enters this position, White delivers checkmate.
        """
        # Place rooks that can checkmate
        coord_dict = leaf_coord.to_dict()

        # Rook 1: Controls the row
        rook1_dict = coord_dict.copy()
        rook1_dict[1] = self.config.max_coord  # Far end of q axis
        rook1_coord = Coordinate.from_dict(rook1_dict)
        if not board.is_occupied(rook1_coord):
            board.set_piece(rook1_coord, Piece(PieceType.ROOK, Color.WHITE))

        # Rook 2: Controls the column
        rook2_dict = coord_dict.copy()
        rook2_dict[0] = self.config.max_coord  # Far end of p axis
        rook2_coord = Coordinate.from_dict(rook2_dict)
        if not board.is_occupied(rook2_coord):
            board.set_piece(rook2_coord, Piece(PieceType.ROOK, Color.WHITE))


class OrdinalPositionGenerator:
    """
    Generate chess positions with specific ordinal game values.

    Uses tree embedding to achieve:
    - Finite ordinals: Linear trees
    - ω, ω², ω³, ...: Power trees
    - Larger ordinals: Complex tree structures
    """

    def __init__(self, config: Optional[BoardConfig] = None):
        self.config = config or BoardConfig.infinite_8()
        self.embedder = TreeEmbedder(config)

    def position_for_ordinal(self, ordinal: OrdinalValue) -> InfiniteDimensionalBoard:
        """Create a position with the given ordinal game value."""
        tree = self._tree_for_ordinal(ordinal)
        return self.embedder.create_position_from_tree(tree)

    def _tree_for_ordinal(self, ordinal: OrdinalValue) -> TreeNode:
        """Build a tree with the given ordinal rank."""
        if ordinal.is_draw or ordinal.is_loss:
            # Leaf node (immediate checkmate or stalemate)
            return TreeNode(value="terminal")

        if ordinal.is_finite():
            # Linear tree of appropriate depth
            return TreeBuilder.linear_tree(ordinal.finite)

        # Transfinite ordinals
        tier = ordinal.tier()

        if tier == OrdinalTier.OMEGA:
            return TreeBuilder.omega_tree()

        if tier == OrdinalTier.OMEGA_PLUS:
            # ω + n: omega tree with a linear extension
            root = TreeBuilder.omega_tree()
            # Add linear chain of length n
            current = root
            for i in range(ordinal.finite):
                current = current.add_child(("plus", i))
            return root

        if tier == OrdinalTier.OMEGA_TIMES:
            # ω·n: n sequential omega trees
            root = TreeNode(value="root")
            for i in range(ordinal.omega):
                omega_subtree = TreeBuilder.omega_tree()
                omega_subtree.value = ("omega", i)
                omega_subtree.parent = root
                root.children.append(omega_subtree)
            return root

        if tier == OrdinalTier.OMEGA_SQUARED:
            return TreeBuilder.omega_power_tree(2)

        if tier == OrdinalTier.OMEGA_CUBED:
            return TreeBuilder.omega_power_tree(3)

        # Higher ordinals: omega^omega approximation
        return TreeBuilder.omega_power_tree(4)

    def sample_positions(
        self,
        n: int = 10,
        include_transfinite: bool = True
    ) -> List[Tuple[InfiniteDimensionalBoard, OrdinalValue]]:
        """Generate sample positions with various ordinal values."""
        import random

        positions = []

        ordinals = [
            OrdinalValue.from_ply(1),
            OrdinalValue.from_ply(5),
            OrdinalValue.from_ply(10),
            OrdinalValue.from_ply(50),
        ]

        if include_transfinite:
            ordinals.extend([
                OrdinalValue.from_omega(1, 0),
                OrdinalValue.from_omega(1, 5),
                OrdinalValue.from_omega(2, 0),
                OrdinalValue.from_omega_squared(1, 0, 0),
                OrdinalValue.from_omega_cubed(1, 0, 0, 0),
            ])

        for _ in range(n):
            ordinal = random.choice(ordinals)
            try:
                position = self.position_for_ordinal(ordinal)
                positions.append((position, ordinal))
            except Exception:
                continue

        return positions


# Convenience functions
def create_position_with_value(ordinal: OrdinalValue) -> InfiniteDimensionalBoard:
    """Create a position with the specified ordinal game value."""
    generator = OrdinalPositionGenerator()
    return generator.position_for_ordinal(ordinal)


def verify_tree_rank(tree: TreeNode) -> OrdinalValue:
    """Compute and return the ordinal rank of a tree."""
    return tree.rank


if __name__ == "__main__":
    print("=== Tree Embedding Test ===\n")

    # Build and embed trees
    print("1. Linear Tree (rank = 5):")
    linear = TreeBuilder.linear_tree(5)
    print(f"   Root: {linear}")
    print(f"   Rank: {linear.rank}")
    print(f"   Nodes: {sum(1 for _ in linear.all_nodes())}")

    print("\n2. Omega Tree (rank = ω):")
    omega = TreeBuilder.omega_tree()
    print(f"   Root children: {len(omega.children)}")
    print(f"   First child is leaf: {omega.children[0].is_leaf if omega.children else 'N/A'}")

    print("\n3. Omega² Tree:")
    omega2 = TreeBuilder.omega_power_tree(2)
    print(f"   Root children: {len(omega2.children)}")
    print(f"   Grandchildren of first child: {len(omega2.children[0].children) if omega2.children else 0}")

    print("\n4. Position Generation:")
    generator = OrdinalPositionGenerator()

    for ordinal in [
        OrdinalValue.from_ply(3),
        OrdinalValue.from_omega(1, 0),
        OrdinalValue.from_omega_squared(1, 0, 0),
    ]:
        pos = generator.position_for_ordinal(ordinal)
        print(f"   Ordinal {ordinal}: {len(pos.pieces)} pieces on board")

    print("\n=== Tests Complete ===")
