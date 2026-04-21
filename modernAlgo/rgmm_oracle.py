from __future__ import annotations

from typing import Dict, Hashable, Iterable, List, Optional, Protocol, Set, Tuple

from modernAlgo.ranks import LazyEdgeRanks, canonical_edge

Node = Hashable
Edge = Tuple[Node, Node]


class GraphView(Protocol):
    """
    Protocol for graph views used by the RGMM oracle.

    Any graph view plugged into the oracle must implement:

    - incident_edges(v):
        iterable of all undirected edges incident to v in the view

    For lazy copied graphs, the oracle also supports an optional indexed
    adjacency API:

    - degree(v):
        number of incident edges

    - incident_edge_at(v, i):
        i-th incident edge, for 0 <= i < degree(v)

    Notes
    -----
    Edges are treated as undirected.
    The view may represent:
    - the original graph
    - an induced subgraph
    - a copied graph for b-matching
    - any other derived graph
    """

    def incident_edges(self, v: Node) -> Iterable[Edge]:
        ...


def edge_endpoints(edge: Edge) -> tuple[Node, Node]:
    """
    Convenience helper.
    """
    return edge[0], edge[1]


class RGMMOracle:
    """
    Recursive local oracle for Random Greedy Maximal Matching (RGMM).

    This implements the standard edge-oracle / vertex-oracle view:

    - EO(e): is edge e in GMM(G, pi)?
    - VO(v): is vertex v matched in GMM(G, pi)?

    where pi is represented implicitly by lazy random edge ranks.

    Important
    ---------
    This is the *core correctness oracle*.
    It assumes the graph view can enumerate incident edges of a vertex.

    Later, for performance-sensitive sublinear behavior, the graph-view
    implementation may internally simulate induced/copy views without
    explicitly materializing them. But this oracle logic remains the same.
    """

    def __init__(self, graph_view: GraphView, ranks: LazyEdgeRanks) -> None:
        self.graph = graph_view
        self.ranks = ranks

        # Memoization caches
        self._edge_cache: Dict[Edge, bool] = {}
        self._vertex_cache: Dict[Node, bool] = {}

        # Optional instrumentation counters
        self.edge_queries = 0
        self.vertex_queries = 0
        self.edge_cache_hits = 0
        self.vertex_cache_hits = 0

    def reset_caches(self) -> None:
        """
        Clear memoization caches but keep the same lazy rank assignment.
        """
        self._edge_cache.clear()
        self._vertex_cache.clear()

        self.edge_queries = 0
        self.vertex_queries = 0
        self.edge_cache_hits = 0
        self.vertex_cache_hits = 0

    def _incident_edges(self, vertex: Node) -> Iterable[Edge]:
        """
        Iterate over incident edges, preferring indexed lazy access when present.
        """
        degree = getattr(self.graph, "degree", None)
        incident_edge_at = getattr(self.graph, "incident_edge_at", None)

        if callable(degree) and callable(incident_edge_at):
            return (
                incident_edge_at(vertex, index)
                for index in range(degree(vertex))
            )

        return self.graph.incident_edges(vertex)

    def _lower_rank_neighboring_edges(self, edge: Edge) -> List[Edge]:
        """
        Return all edges incident to either endpoint of `edge` that appear
        before `edge` in the lazy random order.

        The edge itself is excluded.
        """
        e = canonical_edge(edge)
        u, v = edge_endpoints(e)

        neighbors: Set[Edge] = set()

        for incident in self._incident_edges(u):
            inc = canonical_edge(incident)
            if inc != e and self.ranks.is_before(inc, e):
                neighbors.add(inc)

        for incident in self._incident_edges(v):
            inc = canonical_edge(incident)
            if inc != e and self.ranks.is_before(inc, e):
                neighbors.add(inc)

        # Deterministic processing order
        return sorted(neighbors, key=self.ranks.order_key)

    def edge_in_matching(self, edge: Edge) -> bool:
        """
        Edge oracle EO(e):
        Return True iff `edge` belongs to the RGMM output under the lazy order.

        Logic:
        - An edge is in the greedy matching iff no lower-ranked neighboring edge
          is itself in the matching.
        """
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
        """
        Vertex oracle VO(v):
        Return True iff `vertex` is matched in the RGMM output.
        """
        self.vertex_queries += 1

        if vertex in self._vertex_cache:
            self.vertex_cache_hits += 1
            return self._vertex_cache[vertex]

        incident = sorted(
            (canonical_edge(e) for e in self._incident_edges(vertex)),
            key=self.ranks.order_key,
        )

        for edge in incident:
            if self.edge_in_matching(edge):
                self._vertex_cache[vertex] = True
                return True

        self._vertex_cache[vertex] = False
        return False

    def matched_edge(self, vertex: Node) -> Optional[Edge]:
        """
        Return the unique matched edge incident to `vertex`, if one exists.

        Returns
        -------
        Edge | None
            The matched edge touching `vertex`, or None if `vertex` is unmatched.
        """
        incident = sorted(
            (canonical_edge(e) for e in self._incident_edges(vertex)),
            key=self.ranks.order_key,
        )

        for edge in incident:
            if self.edge_in_matching(edge):
                return edge

        return None

    def matched_vertices_fraction(self, sample_vertices: Iterable[Node]) -> float:
        """
        Convenience utility:
        Return the fraction of sampled vertices that are matched.
        """
        sample_list = list(sample_vertices)
        if not sample_list:
            return 0.0

        matched_count = sum(1 for v in sample_list if self.vertex_matched(v))
        return matched_count / len(sample_list)
