from __future__ import annotations

from collections import defaultdict
from typing import Dict, Hashable, Iterable, List, Protocol, Sequence, Tuple

Node = Hashable
Edge = Tuple[Node, Node]


class BipartiteGraphOracle(Protocol):
    """
    Adjacency-list access interface for a bipartite graph.
    """

    def left_vertices(self) -> Sequence[Node]:
        ...

    def right_vertices(self) -> Sequence[Node]:
        ...

    def vertices(self) -> Sequence[Node]:
        ...

    def num_vertices(self) -> int:
        ...

    def degree(self, v: Node) -> int:
        ...

    def neighbor_at(self, v: Node, index: int) -> Node:
        ...

    def side(self, v: Node) -> str:
        ...


class EdgeListBipartiteGraph:
    """
    Adjacency-list oracle wrapper for the project's existing edge-list graphs.

    This keeps the experiment generators usable while letting the algorithm
    consume graph-oracle operations. Edges are stored in left-to-right direction
    for testing, reporting, and exact baselines.
    """

    def __init__(
        self,
        left_nodes: Sequence[Node],
        right_nodes: Sequence[Node],
        edges: Iterable[Edge],
    ) -> None:
        self._left = list(left_nodes)
        self._right = list(right_nodes)
        self._vertices = self._left + self._right
        self._left_set = set(self._left)
        self._right_set = set(self._right)

        overlap = self._left_set & self._right_set
        if overlap:
            raise ValueError("left and right partitions must be disjoint")

        self._adjacency: Dict[Node, List[Node]] = defaultdict(list)
        for v in self._vertices:
            self._adjacency[v] = []

        self._edges: List[Edge] = []
        seen_edges: set[Edge] = set()

        for a, b in edges:
            u, v = self._orient_edge(a, b)
            edge = (u, v)
            if edge in seen_edges:
                continue

            seen_edges.add(edge)
            self._edges.append(edge)
            self._adjacency[u].append(v)
            self._adjacency[v].append(u)

    def _orient_edge(self, a: Node, b: Node) -> Edge:
        if a in self._left_set and b in self._right_set:
            return a, b
        if b in self._left_set and a in self._right_set:
            return b, a

        raise ValueError(f"edge {a!r}, {b!r} does not cross the bipartition")

    def left_vertices(self) -> Sequence[Node]:
        return self._left

    def right_vertices(self) -> Sequence[Node]:
        return self._right

    def vertices(self) -> Sequence[Node]:
        return self._vertices

    def edges(self) -> Sequence[Edge]:
        return self._edges

    def num_vertices(self) -> int:
        return len(self._vertices)

    def degree(self, v: Node) -> int:
        return len(self._adjacency.get(v, []))

    def neighbor_at(self, v: Node, index: int) -> Node:
        neighbors = self._adjacency.get(v)
        if neighbors is None or index < 0 or index >= len(neighbors):
            raise IndexError("neighbor index out of range")

        return neighbors[index]

    def side(self, v: Node) -> str:
        if v in self._left_set:
            return "L"
        if v in self._right_set:
            return "R"

        raise KeyError(f"unknown vertex: {v!r}")
