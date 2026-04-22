from __future__ import annotations

import math
import random
from typing import Dict, Hashable, Iterable, List, Set, Tuple

from modernAlgo.graph_oracle import BipartiteGraphOracle

Node = Hashable
OriginalEdge = Tuple[Node, Node]
CopiedNode = Tuple[str, Node, int]


def matched_vertices(matching: Iterable[OriginalEdge]) -> Set[Node]:
    """
    Return the set of matched vertices from a matching.
    """
    matched: Set[Node] = set()
    for u, v in matching:
        matched.add(u)
        matched.add(v)
    return matched


class Case2CopiedView:
    """
    Lazy copied-graph view for Case 2.

    This represents the copied bipartite graph G2' from Algorithm 5:
    - vertices in V(M) get k copies
    - vertices in V \\ V(M) get ceil(k*b) copies
    - each original edge in G2 induces all edges between the relevant copies

    Copied neighbors are generated on demand from the original graph oracle.
    The view does not materialize G2 edges or copied edges.
    """

    def __init__(
        self,
        graph: BipartiteGraphOracle,
        M: Iterable[OriginalEdge] = (),
        k: int = 10,
        b: float | None = None,
    ) -> None:
        if k <= 0:
            raise ValueError("k must be positive")

        if b is None:
            b = 1.0 + math.sqrt(2.0)

        self.graph = graph
        self.left_nodes = list(graph.left_vertices())
        self.right_nodes = list(graph.right_vertices())
        self.k = k
        self.b = b
        self.kb = math.ceil(k * b)

        self._matched = matched_vertices(M)

        # Partition original vertices by matched status
        self.matched_left = [u for u in self.left_nodes if u in self._matched]
        self.unmatched_left = [u for u in self.left_nodes if u not in self._matched]
        self.matched_right = [v for v in self.right_nodes if v in self._matched]
        self.unmatched_right = [v for v in self.right_nodes if v not in self._matched]

        self._copy_counts: Dict[Tuple[str, Node], int] = {}
        for u in self.matched_left:
            self._copy_counts[("L", u)] = self.k
        for v in self.matched_right:
            self._copy_counts[("R", v)] = self.k
        for u in self.unmatched_left:
            self._copy_counts[("L", u)] = self.kb
        for v in self.unmatched_right:
            self._copy_counts[("R", v)] = self.kb

        self._num_vertices = (
            self.k * (len(self.matched_left) + len(self.matched_right))
            + self.kb * (len(self.unmatched_left) + len(self.unmatched_right))
        )
        self._degree_cache: Dict[CopiedNode, int] = {}

    def _copied_vertex_at(self, index: int) -> CopiedNode:
        if index < 0 or index >= self._num_vertices:
            raise IndexError("copied vertex index out of range")

        groups = (
            ("L", self.matched_left, self.k),
            ("R", self.matched_right, self.k),
            ("L", self.unmatched_left, self.kb),
            ("R", self.unmatched_right, self.kb),
        )

        offset = index
        for side, nodes, copies in groups:
            group_size = len(nodes) * copies
            if offset < group_size:
                node_index, copy_index = divmod(offset, copies)
                return (side, nodes[node_index], copy_index)
            offset -= group_size

        raise IndexError("copied vertex index out of range")

    def _copy_count(self, side: str, node: Node) -> int:
        return self._copy_counts.get((side, node), 0)

    def _original_copy_count(self, node: Node) -> int:
        return self._copy_count(self.graph.side(node), node)

    def _valid_original_neighbor(self, node: Node, neighbor: Node) -> bool:
        return self._original_copy_count(neighbor) > 0 and (
            (node in self._matched) != (neighbor in self._matched)
        )

    def _valid_copied_vertex(self, v: CopiedNode) -> bool:
        side, node, copy_index = v
        if side not in {"L", "R"}:
            return False

        return 0 <= copy_index < self._copy_count(side, node)

    def sample_vertices(self, num_samples: int, seed: int | None = None) -> List[CopiedNode]:
        """
        Sample copied vertices uniformly with replacement without materializing all vertices.
        """
        if self._num_vertices == 0 or num_samples <= 0:
            return []

        rng = random.Random(seed)
        return [
            self._copied_vertex_at(rng.randrange(self._num_vertices))
            for _ in range(num_samples)
        ]

    def _copied_degree(self, v: CopiedNode) -> int:
        """
        Return the degree of copied vertex v in G2' without materializing its edges.
        """
        if not self._valid_copied_vertex(v):
            return 0

        if v not in self._degree_cache:
            _, node, _ = v
            degree = 0
            for index in range(self.graph.degree(node)):
                neighbor = self.graph.neighbor_at(node, index)
                if self._valid_original_neighbor(node, neighbor):
                    degree += self._original_copy_count(neighbor)

            self._degree_cache[v] = degree

        return self._degree_cache[v]

    def degree(self, v: CopiedNode) -> int:
        """
        Return the degree of copied vertex v in G2'.
        """
        return self._copied_degree(v)

    def neighbor_at(self, v: CopiedNode, index: int) -> CopiedNode:
        """
        Return the index-th copied neighbor of v without materializing all neighbors.
        """
        degree = self._copied_degree(v)
        if index < 0 or index >= degree:
            raise IndexError("copied neighbor index out of range")

        _, node, _ = v
        offset = index

        for neighbor_index in range(self.graph.degree(node)):
            neighbor = self.graph.neighbor_at(node, neighbor_index)
            if not self._valid_original_neighbor(node, neighbor):
                continue

            other_count = self._original_copy_count(neighbor)
            if offset < other_count:
                return (self.graph.side(neighbor), neighbor, offset)

            offset -= other_count

        raise RuntimeError("copied neighbor offset exceeded copied degree")

    def num_vertices(self) -> int:
        return self._num_vertices
