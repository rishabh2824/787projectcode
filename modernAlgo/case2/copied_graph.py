from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, Hashable, Iterable, List, Sequence, Set, Tuple

from modernAlgo.graph_oracle import BipartiteGraphOracle, EdgeListBipartiteGraph

Node = Hashable
OriginalEdge = Tuple[Node, Node]
CopiedNode = Tuple[str, Node, int]
CopiedEdge = Tuple[CopiedNode, CopiedNode]


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

    The original G2 edge set is stored, but copied vertices and copied edges
    are generated on demand instead of being materialized into adjacency lists.
    """

    def __init__(
        self,
        left_nodes: Sequence[Node] | None = None,
        right_nodes: Sequence[Node] | None = None,
        edges: Iterable[OriginalEdge] | None = None,
        M: Iterable[OriginalEdge] = (),
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
        self.original_edges = list(self._iter_left_edges())

        self.k = k
        self.b = b
        self.kb = math.ceil(k * b)

        self._matched = matched_vertices(M)

        # Partition original vertices by matched status
        self.matched_left = [u for u in self.left_nodes if u in self._matched]
        self.unmatched_left = [u for u in self.left_nodes if u not in self._matched]
        self.matched_right = [v for v in self.right_nodes if v in self._matched]
        self.unmatched_right = [v for v in self.right_nodes if v not in self._matched]

        matched_left_set = set(self.matched_left)
        unmatched_left_set = set(self.unmatched_left)
        matched_right_set = set(self.matched_right)
        unmatched_right_set = set(self.unmatched_right)

        # Original G2 edges:
        #   (matched_left, unmatched_right) OR (unmatched_left, matched_right)
        self.g2_edges: List[OriginalEdge] = [
            (u, v)
            for (u, v) in self.original_edges
            if (u in matched_left_set and v in unmatched_right_set)
            or (u in unmatched_left_set and v in matched_right_set)
        ]

        self._copy_counts: Dict[Tuple[str, Node], int] = {}
        for u in self.matched_left:
            self._copy_counts[("L", u)] = self.k
        for v in self.matched_right:
            self._copy_counts[("R", v)] = self.k
        for u in self.unmatched_left:
            self._copy_counts[("L", u)] = self.kb
        for v in self.unmatched_right:
            self._copy_counts[("R", v)] = self.kb

        self._original_adj: Dict[Tuple[str, Node], List[OriginalEdge]] = defaultdict(list)
        for u, v in self.g2_edges:
            self._original_adj[("L", u)].append((u, v))
            self._original_adj[("R", v)].append((u, v))

        self._num_vertices = (
            self.k * (len(self.matched_left) + len(self.matched_right))
            + self.kb * (len(self.unmatched_left) + len(self.unmatched_right))
        )
        self._num_edges = sum(
            self._copy_counts[("L", u)] * self._copy_counts[("R", v)]
            for u, v in self.g2_edges
        )

    def _iter_left_edges(self) -> Iterable[OriginalEdge]:
        for u in self.left_nodes:
            for index in range(self.graph.degree(u)):
                v = self.graph.neighbor_at(u, index)
                yield (u, v)

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
        Return the degree of copied vertex v in G2' without materializing its edges.
        """
        if not self._valid_copied_vertex(v):
            return 0

        side, node, _ = v
        if side == "L":
            return sum(
                self._copy_count("R", original_v)
                for _, original_v in self._original_adj.get((side, node), [])
            )

        return sum(
            self._copy_count("L", u)
            for u, _ in self._original_adj.get((side, node), [])
        )

    def degree(self, v: CopiedNode) -> int:
        """
        Return the degree of copied vertex v in G2'.
        """
        return self.copied_degree(v)

    def neighbor_at(self, v: CopiedNode, index: int) -> CopiedNode:
        """
        Return the index-th copied neighbor of v without materializing all neighbors.
        """
        degree = self.copied_degree(v)
        if index < 0 or index >= degree:
            raise IndexError("copied neighbor index out of range")

        side, node, _ = v
        offset = index

        for u, original_v in self._original_adj.get((side, node), []):
            if side == "L":
                other_count = self._copy_count("R", original_v)
                if offset < other_count:
                    return ("R", original_v, offset)
            else:
                other_count = self._copy_count("L", u)
                if offset < other_count:
                    return ("L", u, offset)

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
        Sample a uniformly random copied neighbor of v in G2'.
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
        Sample a uniformly random copied edge incident to v in G2'.
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

        side, node, copy_index = v
        edges: List[CopiedEdge] = []
        for u, original_v in self._original_adj.get((side, node), []):
            if side == "L":
                other_count = self._copy_count("R", original_v)
                left_copy = ("L", u, copy_index)
                for j in range(other_count):
                    edges.append((left_copy, ("R", original_v, j)))
            else:
                other_count = self._copy_count("L", u)
                right_copy = ("R", original_v, copy_index)
                for i in range(other_count):
                    edges.append((("L", u, i), right_copy))

        return edges

    def num_vertices(self) -> int:
        return self._num_vertices

    def num_edges(self) -> int:
        return self._num_edges
