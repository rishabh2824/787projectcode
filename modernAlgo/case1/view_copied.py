from __future__ import annotations

import math
import random
from typing import Dict, Hashable, Iterable, List, Sequence, Set, Tuple

from modernAlgo.graph_oracle import BipartiteGraphOracle, EdgeListBipartiteGraph

Node = Hashable
OriginalEdge = Tuple[Node, Node]
CopiedNode = Tuple[str, Node, int]
CopiedEdge = Tuple[CopiedNode, CopiedNode]


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

    Important
    ---------
    In the full oracle implementation, membership in V(M') is determined
    by the *inner oracle* on G[V \\ V(M)], not by explicitly constructing M'.
    This file therefore accepts an oracle-like object supporting:

        vertex_matched(v) -> bool

    for vertices in the unmatched induced subgraph.

    Copied neighbors are generated on demand from the original graph oracle.
    The view does not materialize G1 edges or copied edges.
    """

    def __init__(
        self,
        left_nodes: Sequence[Node] | None = None,
        right_nodes: Sequence[Node] | None = None,
        edges: Iterable[OriginalEdge] | None = None,
        M: Iterable[OriginalEdge] = (),
        inner_mprime_oracle=None,
        k: int = 10,
        b: float | None = None,
        graph: BipartiteGraphOracle | None = None,
    ) -> None:
        if k <= 0:
            raise ValueError("k must be positive")

        if b is None:
            b = 1.0 + math.sqrt(2.0)

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
        self.k = k
        self.b = b
        self.kb = math.ceil(k * b)

        self._matched_M = matched_vertices(M)
        self._inner_oracle = inner_mprime_oracle
        if self._inner_oracle is None:
            raise ValueError("inner_mprime_oracle is required for Case1CopiedView")
        self._mprime_status: Dict[Node, bool] = {}

        # Step 1: classify vertices
        #
        # A = V(M')
        # B = V \ V(M) \ V(M')
        #
        # Only vertices not already matched in explicit M are candidates
        # for either A or B.
        self.A_left: List[Node] = []
        self.A_right: List[Node] = []
        self.B_left: List[Node] = []
        self.B_right: List[Node] = []

        for u in self.left_nodes:
            if u in self._matched_M:
                continue
            if self._is_mprime_matched(u):
                self.A_left.append(u)
            else:
                self.B_left.append(u)

        for v in self.right_nodes:
            if v in self._matched_M:
                continue
            if self._is_mprime_matched(v):
                self.A_right.append(v)
            else:
                self.B_right.append(v)

        self._copy_counts: Dict[Tuple[str, Node], int] = {}
        for u in self.A_left:
            self._copy_counts[("L", u)] = self.k
        for v in self.A_right:
            self._copy_counts[("R", v)] = self.k
        for u in self.B_left:
            self._copy_counts[("L", u)] = self.kb
        for v in self.B_right:
            self._copy_counts[("R", v)] = self.kb

        self._num_vertices = (
            self.k * (len(self.A_left) + len(self.A_right))
            + self.kb * (len(self.B_left) + len(self.B_right))
        )
        self._degree_cache: Dict[CopiedNode, int] = {}
        self._num_edges: int | None = None

    def _copied_vertex_at(self, index: int) -> CopiedNode:
        if index < 0 or index >= self._num_vertices:
            raise IndexError("copied vertex index out of range")

        groups = (
            ("L", self.A_left, self.k),
            ("R", self.A_right, self.k),
            ("L", self.B_left, self.kb),
            ("R", self.B_right, self.kb),
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

        return 0 <= copy_index < self._copy_count(side, node)

    def vertex_at(self, index: int) -> CopiedNode:
        """
        Return the index-th copied vertex without materializing all vertices.
        """
        return self._copied_vertex_at(index)

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

    def copied_degree(self, v: CopiedNode) -> int:
        """
        Return the degree of copied vertex v in G1' without materializing its edges.
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
        Return the degree of copied vertex v in G1'.
        """
        return self.copied_degree(v)

    def neighbor_at(self, v: CopiedNode, index: int) -> CopiedNode:
        """
        Return the index-th copied neighbor of v without materializing all neighbors.
        """
        degree = self.copied_degree(v)
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

    def incident_edge_at(self, v: CopiedNode, index: int) -> CopiedEdge:
        """
        Return the index-th copied edge incident to v without materializing all edges.
        """
        neighbor = self.neighbor_at(v, index)
        if v[0] == "L":
            return v, neighbor

        return neighbor, v

    def random_neighbor(
        self,
        v: CopiedNode,
        rng: random.Random | None = None,
    ) -> CopiedNode | None:
        """
        Sample a uniformly random copied neighbor of v in G1'.
        """
        if rng is None:
            rng = random.Random()

        degree = self.copied_degree(v)
        if degree == 0:
            return None

        return self.neighbor_at(v, rng.randrange(degree))

    def random_incident_edge(
        self,
        v: CopiedNode,
        rng: random.Random | None = None,
    ) -> CopiedEdge | None:
        """
        Sample a uniformly random copied edge incident to v in G1'.
        """
        neighbor = self.random_neighbor(v, rng)
        if neighbor is None:
            return None

        return (v, neighbor) if v[0] == "L" else (neighbor, v)

    def incident_edges(self, v: CopiedNode) -> List[CopiedEdge]:
        """
        Return copied edges incident to copied vertex v.
        """
        if not self._valid_copied_vertex(v):
            return []

        edges: List[CopiedEdge] = []
        for index in range(self.copied_degree(v)):
            neighbor = self.neighbor_at(v, index)
            edges.append((v, neighbor) if v[0] == "L" else (neighbor, v))

        return edges

    def num_vertices(self) -> int:
        return self._num_vertices

    def num_edges(self) -> int:
        if self._num_edges is None:
            total = 0
            for u in self.left_nodes:
                left_count = self._copy_count("L", u)
                if left_count == 0:
                    continue
                for index in range(self.graph.degree(u)):
                    v = self.graph.neighbor_at(u, index)
                    if self._valid_original_neighbor(u, v):
                        total += left_count * self._copy_count("R", v)

            self._num_edges = total

        return self._num_edges
