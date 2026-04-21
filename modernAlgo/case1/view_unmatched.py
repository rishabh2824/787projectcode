from __future__ import annotations

import random
from typing import Dict, Hashable, Iterable, List, Sequence, Set, Tuple

from modernAlgo.graph_oracle import BipartiteGraphOracle, EdgeListBipartiteGraph
from modernAlgo.ranks import canonical_edge

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

    The view keeps original adjacency and explicit membership in V(M), but it
    does not materialize induced edges. Random incident edges are obtained by
    rejection sampling from the original graph adjacency list, matching the
    paper's access pattern for induced subgraphs.
    """

    def __init__(
        self,
        left_nodes: Sequence[Node] | None = None,
        right_nodes: Sequence[Node] | None = None,
        edges: Iterable[Edge] | None = None,
        M: Iterable[Edge] = (),
        graph: BipartiteGraphOracle | None = None,
    ) -> None:
        if graph is None:
            if left_nodes is None or right_nodes is None or edges is None:
                raise ValueError("provide either graph or left_nodes/right_nodes/edges")
            graph = EdgeListBipartiteGraph(left_nodes, right_nodes, edges)

        self.graph = graph
        self.left_nodes = (
            list(graph.left_vertices()) if left_nodes is None else list(left_nodes)
        )
        self.right_nodes = (
            list(graph.right_vertices()) if right_nodes is None else list(right_nodes)
        )

        self._matched = matched_vertices(M)
        self._unmatched_vertices: List[Node] = [
            v for v in (self.left_nodes + self.right_nodes) if v not in self._matched
        ]
        self._unmatched_set: Set[Node] = set(self._unmatched_vertices)
        self._induced_neighbors_cache: Dict[Node, List[Node]] = {}
        self._num_edges: int | None = None

    def is_unmatched_vertex(self, v: Node) -> bool:
        return v in self._unmatched_set

    def sample_vertices(self, num_samples: int, seed: int | None = None) -> List[Node]:
        """
        Sample vertices uniformly from V \\ V(M), with replacement.
        """
        if not self._unmatched_vertices or num_samples <= 0:
            return []

        rng = random.Random(seed)
        return [rng.choice(self._unmatched_vertices) for _ in range(num_samples)]

    def vertex_at(self, index: int) -> Node:
        """
        Return the index-th vertex in V \\ V(M).
        """
        if index < 0 or index >= len(self._unmatched_vertices):
            raise IndexError("unmatched vertex index out of range")

        return self._unmatched_vertices[index]

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

    def incident_edge_at(self, v: Node, index: int) -> Edge:
        """
        Return the index-th induced edge incident to v.
        """
        return canonical_edge((v, self.neighbor_at(v, index)))

    def random_neighbor(
        self,
        v: Node,
        rng: random.Random | None = None,
    ) -> Node | None:
        """
        Sample a uniformly random neighbor of v in G[V \\ V(M)].
        """
        if v not in self._unmatched_set:
            return None

        original_degree = self.graph.degree(v)
        if original_degree == 0:
            return None

        if rng is None:
            rng = random.Random()

        induced_degree = self.degree(v)
        if induced_degree == 0:
            return None

        while True:
            candidate = self.graph.neighbor_at(v, rng.randrange(original_degree))
            if candidate in self._unmatched_set:
                return candidate

    def random_incident_edge(
        self,
        v: Node,
        rng: random.Random | None = None,
    ) -> Edge | None:
        """
        Sample a uniformly random edge incident to v in G[V \\ V(M)].
        """
        neighbor = self.random_neighbor(v, rng)
        if neighbor is None:
            return None

        return canonical_edge((v, neighbor))

    def num_vertices(self) -> int:
        return len(self._unmatched_vertices)

    def num_edges(self) -> int:
        if self._num_edges is None:
            total_degree = sum(self.degree(v) for v in self._unmatched_vertices)
            self._num_edges = total_degree // 2

        return self._num_edges
