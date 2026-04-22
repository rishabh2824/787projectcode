from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, Hashable, Iterable, Optional, Protocol, Tuple

from modernAlgo.ranks import canonical_edge

Node = Hashable
Edge = Tuple[Node, Node]


class AdjacencyListGraphView(Protocol):
    """
    Protocol needed by the Behnezhad Appendix A oracle.
    """

    def degree(self, v: Node) -> int:
        ...

    def neighbor_at(self, v: Node, index: int) -> Node:
        ...

    def num_vertices(self) -> int:
        ...


class BehnezhadRGMMOracle:
    """
    Appendix-A-style local oracle for random greedy maximal matching.

    This implements the structure from Behnezhad's full version:
    - Algorithm 6: vertex oracle VO(v)
    - Algorithm 7: oriented edge oracle EO(e=(u, v), u)
    - Algorithm 8: lowest(v, i)
    - expose_next(v): lazily exposes rank intervals of v's adjacency list

    The oracle uses adjacency-list graph access and samples edge-rank
    intervals lazily, so copied graphs do not need to be materialized.
    """

    def __init__(
        self,
        graph_view: AdjacencyListGraphView,
        seed: int | None = None,
        max_degree_bound: int | None = None,
    ) -> None:
        self.graph = graph_view
        self.rng = random.Random(seed)
        self.max_degree_bound = self._choose_degree_bound(max_degree_bound)
        self.intervals = self._build_intervals(self.max_degree_bound)

        self._edge_interval: Dict[Edge, int] = {}
        self._exposed_by_node: Dict[Node, Dict[Node, float]] = defaultdict(dict)
        self._k: Dict[Node, int] = defaultdict(int)

        self._oriented_edge_cache: Dict[tuple[Edge, Node], bool] = {}
        self._vertex_cache: Dict[Node, bool] = {}
        self._matched_edge_cache: Dict[Node, Optional[Edge]] = {}

    def _choose_degree_bound(self, max_degree_bound: int | None) -> int:
        if max_degree_bound is not None:
            if max_degree_bound <= 0:
                raise ValueError("max_degree_bound must be positive")
            bound = max_degree_bound
        else:
            # Any upper bound on the maximum degree is valid for the interval
            # scheme. num_vertices() is always a safe upper bound in a simple
            # graph view and avoids materializing copied vertices.
            bound = max(1, self.graph.num_vertices())

        return 1 << (bound - 1).bit_length()

    def _build_intervals(self, degree_bound: int) -> list[tuple[float, float]]:
        log_delta = int(math.log2(degree_bound))
        intervals: list[tuple[float, float]] = []

        for i in range(log_delta + 1):
            if i == 0:
                start = 0.0
                end = 1.0 / degree_bound
            else:
                start = (2 ** (i - 1)) / degree_bound
                end = (2**i) / degree_bound

            intervals.append((start, min(end, 1.0)))

        return intervals

    def _sample_indices(self, n: int, probability: float) -> Iterable[int]:
        """
        Sample indices from range(n), independently with the given probability.
        """
        if probability <= 0.0 or n <= 0:
            return []
        if probability >= 1.0:
            return range(n)

        def generator() -> Iterable[int]:
            index = -1
            log_miss = math.log1p(-probability)
            while True:
                draw = self.rng.random()
                while draw == 0.0:
                    draw = self.rng.random()
                skip = math.floor(math.log(draw) / log_miss)
                index += skip + 1
                if index >= n:
                    break
                yield index

        return generator()

    def _exposed_ready_count(self, v: Node) -> int:
        k_v = self._k[v]
        return sum(
            1
            for u in self._exposed_by_node[v]
            if self._edge_interval[canonical_edge((v, u))] < k_v
        )

    def _exposed_sorted(self, v: Node) -> list[tuple[float, Edge, Node]]:
        return sorted(
            (
                (rank, canonical_edge((v, u)), u)
                for u, rank in self._exposed_by_node[v].items()
            ),
            key=lambda item: (item[0], item[1]),
        )

    def _expose_edge(self, v: Node, u: Node, interval_index: int, rank: float) -> None:
        edge = canonical_edge((v, u))
        self._edge_interval[edge] = interval_index
        self._exposed_by_node[v][u] = rank
        self._exposed_by_node[u][v] = rank

    def expose_next(self, v: Node) -> None:
        """
        Expose all sampled edges of v in its current rank interval.
        """
        current_k = self._k[v]
        if current_k >= len(self.intervals):
            return

        start, end = self.intervals[current_k]
        probability = (end - start) / (1.0 - start)
        degree = self.graph.degree(v)

        for index in self._sample_indices(degree, probability):
            u = self.graph.neighbor_at(v, index)

            if u in self._exposed_by_node[v]:
                continue

            if self._k[u] <= current_k:
                rank = self.rng.uniform(start, end)
                self._expose_edge(v, u, current_k, rank)

        self._k[v] += 1

    def lowest(self, v: Node, index: int) -> Node | None:
        """
        Return the 1-indexed index-th lowest-rank neighbor of v.
        """
        if index <= 0:
            raise ValueError("lowest index must be positive")
        if index > self.graph.degree(v):
            return None

        while self._exposed_ready_count(v) < index:
            previous_k = self._k[v]
            self.expose_next(v)
            if self._k[v] == previous_k:
                return None

        exposed = self._exposed_sorted(v)
        if index > len(exposed):
            return None

        return exposed[index - 1][2]

    def _oriented_edge_available(
        self,
        edge: Edge,
        endpoint: Node,
    ) -> bool:
        """
        Algorithm 7: determine whether endpoint is free before edge is processed.
        """
        e = canonical_edge(edge)
        cache_key = (e, endpoint)

        if cache_key in self._oriented_edge_cache:
            return self._oriented_edge_cache[cache_key]

        other = e[1] if endpoint == e[0] else e[0]
        j = 1
        w = self.lowest(endpoint, j)

        while w is not None and w != other:
            if self._oriented_edge_available(
                (endpoint, w),
                w,
            ):
                self._oriented_edge_cache[cache_key] = False
                return False

            j += 1
            w = self.lowest(endpoint, j)

        self._oriented_edge_cache[cache_key] = True
        return True

    def matched_edge(self, vertex: Node) -> Optional[Edge]:
        """
        Return the unique RGMM edge incident to vertex, if one exists.
        """
        if vertex in self._matched_edge_cache:
            return self._matched_edge_cache[vertex]

        degree = self.graph.degree(vertex)
        for j in range(1, degree + 1):
            w = self.lowest(vertex, j)
            if w is None:
                break

            if self._oriented_edge_available((vertex, w), w):
                edge = canonical_edge((vertex, w))
                self._matched_edge_cache[vertex] = edge
                return edge

        self._matched_edge_cache[vertex] = None
        return None

    def vertex_matched(self, vertex: Node) -> bool:
        """
        Return True iff vertex is matched in the locally simulated RGMM.
        """
        if vertex in self._vertex_cache:
            return self._vertex_cache[vertex]

        matched = self.matched_edge(vertex) is not None
        self._vertex_cache[vertex] = matched
        return matched

    def matched_vertices_fraction(self, sample_vertices: Iterable[Node]) -> float:
        """
        Return the fraction of sampled vertices that are matched.
        """
        sample_list = list(sample_vertices)
        if not sample_list:
            return 0.0

        matched_count = sum(1 for v in sample_list if self.vertex_matched(v))
        return matched_count / len(sample_list)
