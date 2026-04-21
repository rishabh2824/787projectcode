from __future__ import annotations

import math
import random
from typing import Dict, Hashable, Iterable, Optional, Protocol, Set, Tuple

from modernAlgo.ranks import LazyEdgeRanks, canonical_edge

Node = Hashable
Edge = Tuple[Node, Node]


class RandomNeighborGraphView(Protocol):
    """
    Protocol for graph views used by RandomNeighborRGMMOracle.
    """

    def degree(self, v: Node) -> int:
        ...

    def incident_edge_at(self, v: Node, index: int) -> Edge:
        ...

    def random_incident_edge(
        self,
        v: Node,
        rng: random.Random | None = None,
    ) -> Edge | None:
        ...

    def num_vertices(self) -> int:
        ...


def edge_endpoints(edge: Edge) -> tuple[Node, Node]:
    return edge[0], edge[1]


class RandomNeighborRGMMOracle:
    """
    Random-neighbor backed local oracle for random greedy maximal matching.

    This is the copied-graph oracle path. It avoids enumerating high-degree
    copied adjacency lists by sampling incident copied edges. For vertices
    whose copied degree is at most the probe budget, it falls back to exact
    indexed adjacency and matches RGMMOracle's behavior.

    Important
    ---------
    For copied vertices with degree above the probe budget, this is a bounded
    random-neighbor simulation of the RGMM dependency recursion. It is the
    implementation hook needed by the paper-style random-neighbor access model,
    but it is not a full formal reimplementation of Behnezhad's Proposition 3.1
    oracle.
    """

    def __init__(
        self,
        graph_view: RandomNeighborGraphView,
        ranks: LazyEdgeRanks,
        seed: int | None = None,
        probe_budget: int | None = None,
    ) -> None:
        self.graph = graph_view
        self.ranks = ranks
        self.rng = random.Random(seed)
        self.probe_budget = (
            self._default_probe_budget()
            if probe_budget is None
            else max(1, probe_budget)
        )

        self._edge_cache: Dict[Edge, bool] = {}
        self._vertex_cache: Dict[Node, bool] = {}

        self.edge_queries = 0
        self.vertex_queries = 0
        self.edge_cache_hits = 0
        self.vertex_cache_hits = 0
        self.random_edge_probes = 0
        self.exact_low_degree_scans = 0

    def _default_probe_budget(self) -> int:
        n = max(self.graph.num_vertices(), 2)
        return max(1, math.ceil(6 * (math.log(n) ** 3)))

    def reset_caches(self) -> None:
        self._edge_cache.clear()
        self._vertex_cache.clear()

        self.edge_queries = 0
        self.vertex_queries = 0
        self.edge_cache_hits = 0
        self.vertex_cache_hits = 0
        self.random_edge_probes = 0
        self.exact_low_degree_scans = 0

    def _candidate_incident_edges(
        self,
        vertex: Node,
        rank_limit: float | None = None,
        exclude: Edge | None = None,
    ) -> list[Edge]:
        """
        Return candidate incident edges in rank order.

        Low-degree vertices are scanned exactly. High-degree vertices are probed
        using random_incident_edge, deduplicated, filtered, and sorted.
        """
        degree = self.graph.degree(vertex)
        if degree <= 0:
            return []

        excluded = None if exclude is None else canonical_edge(exclude)
        candidates: Set[Edge] = set()

        if degree <= self.probe_budget:
            self.exact_low_degree_scans += 1
            for index in range(degree):
                edge = canonical_edge(self.graph.incident_edge_at(vertex, index))
                if edge == excluded:
                    continue
                if rank_limit is not None and self.ranks.rank(edge) >= rank_limit:
                    continue
                candidates.add(edge)
        else:
            attempts = max(self.probe_budget, min(degree, 4 * self.probe_budget))
            for _ in range(attempts):
                edge = self.graph.random_incident_edge(vertex, self.rng)
                self.random_edge_probes += 1
                if edge is None:
                    continue

                edge = canonical_edge(edge)
                if edge == excluded:
                    continue
                if rank_limit is not None and self.ranks.rank(edge) >= rank_limit:
                    continue
                candidates.add(edge)

                if len(candidates) >= self.probe_budget:
                    break

        return sorted(candidates, key=self.ranks.order_key)

    def _lower_rank_neighboring_edges(self, edge: Edge) -> list[Edge]:
        e = canonical_edge(edge)
        u, v = edge_endpoints(e)
        rank_limit = self.ranks.rank(e)

        neighbors: Set[Edge] = set()
        neighbors.update(
            self._candidate_incident_edges(
                u,
                rank_limit=rank_limit,
                exclude=e,
            )
        )
        neighbors.update(
            self._candidate_incident_edges(
                v,
                rank_limit=rank_limit,
                exclude=e,
            )
        )

        return sorted(neighbors, key=self.ranks.order_key)

    def edge_in_matching(self, edge: Edge) -> bool:
        e = canonical_edge(edge)
        self.edge_queries += 1

        if e in self._edge_cache:
            self.edge_cache_hits += 1
            return self._edge_cache[e]

        for prev_edge in self._lower_rank_neighboring_edges(e):
            if self.edge_in_matching(prev_edge):
                self._edge_cache[e] = False
                return False

        self._edge_cache[e] = True
        return True

    def vertex_matched(self, vertex: Node) -> bool:
        self.vertex_queries += 1

        if vertex in self._vertex_cache:
            self.vertex_cache_hits += 1
            return self._vertex_cache[vertex]

        for edge in self._candidate_incident_edges(vertex):
            if self.edge_in_matching(edge):
                self._vertex_cache[vertex] = True
                return True

        self._vertex_cache[vertex] = False
        return False

    def matched_edge(self, vertex: Node) -> Optional[Edge]:
        for edge in self._candidate_incident_edges(vertex):
            if self.edge_in_matching(edge):
                return edge

        return None

    def matched_vertices_fraction(self, sample_vertices: Iterable[Node]) -> float:
        sample_list = list(sample_vertices)
        if not sample_list:
            return 0.0

        matched_count = sum(1 for v in sample_list if self.vertex_matched(v))
        return matched_count / len(sample_list)
