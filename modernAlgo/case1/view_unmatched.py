from __future__ import annotations

from collections import defaultdict
from typing import Dict, Hashable, Iterable, List, Sequence, Set, Tuple

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
    View of the induced subgraph G[V \\ V(M)].

    This is the graph used for the inner RGMM oracle that defines M'
    in the main paper's bipartite algorithm. Vertices matched in the
    explicit sparsification matching M are removed, and only edges
    between remaining vertices are kept.

    This view supports exactly what RGMMOracle needs:
    - vertices()
    - incident_edges(v)

    We build explicit adjacency for now. This is appropriate for getting
    the oracle logic correct before optimizing further.
    """

    def __init__(
        self,
        left_nodes: Sequence[Node],
        right_nodes: Sequence[Node],
        edges: Iterable[Edge],
        M: Iterable[Edge],
    ) -> None:
        self.left_nodes = list(left_nodes)
        self.right_nodes = list(right_nodes)
        self.edges = [canonical_edge(e) for e in edges]

        self._matched = matched_vertices(M)

        self._vertices: List[Node] = [
            v for v in (self.left_nodes + self.right_nodes) if v not in self._matched
        ]
        self._vertex_set: Set[Node] = set(self._vertices)

        self._adj: Dict[Node, List[Edge]] = defaultdict(list)
        for v in self._vertices:
            self._adj[v] = []

        self._edges: List[Edge] = []
        for e in self.edges:
            u, v = e
            if u in self._vertex_set and v in self._vertex_set:
                self._edges.append(e)
                self._adj[u].append(e)
                self._adj[v].append(e)

    def vertices(self) -> List[Node]:
        """
        Return all vertices in the induced unmatched subgraph.
        """
        return list(self._vertices)

    def incident_edges(self, v: Node) -> List[Edge]:
        """
        Return all edges incident to v in G[V \\ V(M)].
        """
        return list(self._adj.get(v, []))

    def num_vertices(self) -> int:
        return len(self._vertices)

    def num_edges(self) -> int:
        return len(self._edges)
