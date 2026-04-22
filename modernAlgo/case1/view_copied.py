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
    Return the set of matched vertices from an explicit matching.
    """
    matched: Set[Node] = set()
    for u, v in matching:
        matched.add(u)
        matched.add(v)
    return matched


class Case1CopiedView:
    """
    Lazy copied-graph view for Case 1.

    This represents the copied graph G1' used in Algorithm 4:
      G1 = G[V(M'), V \\ V(M) \\ V(M')]

    with:
    - k copies for vertices in V(M')
    - ceil(k*b) copies for vertices in V \\ V(M) \\ V(M')

    The implementation uses a padded copied universe: every vertex in
    V \\ V(M) has ceil(k*b) copy slots, but vertices in V(M') have only their
    first k slots active. The remaining slots are isolated dummy vertices.
    This preserves the matching size while allowing uniform copied-vertex
    sampling without classifying all vertices by their M' status up front.

    Copied neighbors are generated on demand from the original graph oracle.
    The view does not materialize G1 vertices, G1 edges, or copied edges.
    """

    def __init__(
        self,
        graph: BipartiteGraphOracle,
        M: Iterable[OriginalEdge] = (),
        inner_mprime_oracle=None,
        k: int = 10,
        b: float | None = None,
    ) -> None:
        if k <= 0:
            raise ValueError("k must be positive")

        if b is None:
            b = 1.0 + math.sqrt(2.0)

        if inner_mprime_oracle is None:
            raise ValueError("inner_mprime_oracle is required for Case1CopiedView")

        self.graph = graph
        self.k = k
        self.b = b
        self.kb = math.ceil(k * b)

        self._matched_M = matched_vertices(M)
        self._inner_oracle = inner_mprime_oracle
        self._mprime_status: Dict[Node, bool] = {}

        self._unmatched_vertices: List[Node] = [
            v for v in graph.vertices() if v not in self._matched_M
        ]
        self._unmatched_set = set(self._unmatched_vertices)

        self._num_vertices = self.kb * len(self._unmatched_vertices)
        self._actual_copy_count_cache: Dict[Tuple[str, Node], int] = {}
        self._degree_cache: Dict[CopiedNode, int] = {}

    def _copied_vertex_at(self, index: int) -> CopiedNode:
        if index < 0 or index >= self._num_vertices:
            raise IndexError("copied vertex index out of range")

        original_index, copy_index = divmod(index, self.kb)
        node = self._unmatched_vertices[original_index]
        return (self.graph.side(node), node, copy_index)

    def _padded_copy_count(self, side: str, node: Node) -> int:
        if node not in self._unmatched_set:
            return 0
        if side != self.graph.side(node):
            return 0
        return self.kb

    def _copy_count(self, side: str, node: Node) -> int:
        key = (side, node)
        if key not in self._actual_copy_count_cache:
            if node not in self._unmatched_set or side != self.graph.side(node):
                self._actual_copy_count_cache[key] = 0
            elif self._is_mprime_matched(node):
                self._actual_copy_count_cache[key] = self.k
            else:
                self._actual_copy_count_cache[key] = self.kb

        return self._actual_copy_count_cache[key]

    def _original_copy_count(self, node: Node) -> int:
        return self._copy_count(self.graph.side(node), node)

    def _is_mprime_matched(self, node: Node) -> bool:
        if node not in self._mprime_status:
            self._mprime_status[node] = self._inner_oracle.vertex_matched(node)

        return self._mprime_status[node]

    def _valid_original_neighbor(self, node: Node, neighbor: Node) -> bool:
        if self._original_copy_count(neighbor) == 0:
            return False

        return self._is_mprime_matched(node) != self._is_mprime_matched(neighbor)

    def _valid_copied_vertex(self, v: CopiedNode) -> bool:
        side, node, copy_index = v
        if side not in {"L", "R"}:
            return False

        return 0 <= copy_index < self._padded_copy_count(side, node)

    def _active_copied_vertex(self, v: CopiedNode) -> bool:
        side, node, copy_index = v
        return self._valid_copied_vertex(v) and copy_index < self._copy_count(side, node)

    def sample_vertices(self, num_samples: int, seed: int | None = None) -> List[CopiedNode]:
        """
        Sample uniformly from the padded copied universe.

        Dummy isolated copies are included. They never affect the matching
        size, and scaling by the padded universe size remains unbiased for the
        actual copied matching size.
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
        Return the degree of copied vertex v in G1' without materializing edges.
        """
        if not self._active_copied_vertex(v):
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
        Return the degree of copied vertex v in G1'.
        """
        return self._copied_degree(v)

    def neighbor_at(self, v: CopiedNode, index: int) -> CopiedNode:
        """
        Return the index-th copied neighbor of v without materializing neighbors.
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
