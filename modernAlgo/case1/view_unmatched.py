from __future__ import annotations

import random
from typing import Dict, Hashable, Iterable, List, Set, Tuple

from modernAlgo.graph_oracle import BipartiteGraphOracle

Node = Hashable
Edge = Tuple[Node, Node]


def matched_vertices(matching: Iterable[Edge]) -> Set[Node]:
    """
    Return the set of matched vertices from a matching.
    """
    matched: Set[Node] = set()
    for u, v in matching:
        matched.add(u)
        matched.add(v)
    return matched


class UnmatchedInducedView:
    """
    Lazy view of the induced subgraph G[V \\ V(M)].

    The view keeps only explicit membership in V(M). Induced neighbors are
    generated from the original graph oracle and cached per queried vertex.
    """

    def __init__(
        self,
        graph: BipartiteGraphOracle,
        M: Iterable[Edge] = (),
    ) -> None:
        self.graph = graph
        self._matched = matched_vertices(M)
        self._unmatched_vertices: List[Node] = [
            v for v in graph.vertices() if v not in self._matched
        ]
        self._unmatched_set: Set[Node] = set(self._unmatched_vertices)
        self._induced_neighbors_cache: Dict[Node, List[Node]] = {}

    def sample_vertices(self, num_samples: int, seed: int | None = None) -> List[Node]:
        """
        Sample vertices uniformly from V \\ V(M), with replacement.
        """
        if not self._unmatched_vertices or num_samples <= 0:
            return []

        rng = random.Random(seed)
        return [rng.choice(self._unmatched_vertices) for _ in range(num_samples)]

    def _induced_neighbors(self, v: Node) -> List[Node]:
        if v not in self._unmatched_set:
            return []

        if v not in self._induced_neighbors_cache:
            self._induced_neighbors_cache[v] = []
            for index in range(self.graph.degree(v)):
                u = self.graph.neighbor_at(v, index)
                if u in self._unmatched_set:
                    self._induced_neighbors_cache[v].append(u)

        return self._induced_neighbors_cache[v]

    def degree(self, v: Node) -> int:
        """
        Return degree in G[V \\ V(M)].
        """
        return len(self._induced_neighbors(v))

    def neighbor_at(self, v: Node, index: int) -> Node:
        """
        Return the index-th induced neighbor of v.
        """
        neighbors = self._induced_neighbors(v)
        if index < 0 or index >= len(neighbors):
            raise IndexError("unmatched induced neighbor index out of range")

        return neighbors[index]

    def num_vertices(self) -> int:
        return len(self._unmatched_vertices)
