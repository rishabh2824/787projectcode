from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Hashable, Iterable, List, Sequence, Set, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modernAlgo.ranks import canonical_edge

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

    For correctness-first development, we classify vertices up front using:
    - explicit M for V(M)
    - inner oracle for V(M')

    The original G1 edge set is stored, but copied vertices and copied edges
    are generated on demand instead of being materialized into adjacency lists.
    """

    def __init__(
        self,
        left_nodes: Sequence[Node],
        right_nodes: Sequence[Node],
        edges: Iterable[OriginalEdge],
        M: Iterable[OriginalEdge],
        inner_mprime_oracle,
        k: int,
        b: float | None = None,
    ) -> None:
        if k <= 0:
            raise ValueError("k must be positive")

        if b is None:
            b = 1.0 + math.sqrt(2.0)

        self.left_nodes = list(left_nodes)
        self.right_nodes = list(right_nodes)
        self.original_edges = [canonical_edge(e) for e in edges]

        self.k = k
        self.b = b
        self.kb = math.ceil(k * b)

        self._matched_M = matched_vertices(M)
        self._inner_oracle = inner_mprime_oracle

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
            if self._inner_oracle.vertex_matched(u):
                self.A_left.append(u)
            else:
                self.B_left.append(u)

        for v in self.right_nodes:
            if v in self._matched_M:
                continue
            if self._inner_oracle.vertex_matched(v):
                self.A_right.append(v)
            else:
                self.B_right.append(v)

        A_left_set = set(self.A_left)
        A_right_set = set(self.A_right)
        B_left_set = set(self.B_left)
        B_right_set = set(self.B_right)

        # Step 2: build original G1 edges
        #
        # Since original edges are (u in U, v in V), valid G1 edges are:
        #   (A_left, B_right) OR (B_left, A_right)
        self.g1_edges: List[OriginalEdge] = [
            (u, v)
            for (u, v) in self.original_edges
            if (u in A_left_set and v in B_right_set)
            or (u in B_left_set and v in A_right_set)
        ]

        self._copy_counts: Dict[Tuple[str, Node], int] = {}
        for u in self.A_left:
            self._copy_counts[("L", u)] = self.k
        for v in self.A_right:
            self._copy_counts[("R", v)] = self.k
        for u in self.B_left:
            self._copy_counts[("L", u)] = self.kb
        for v in self.B_right:
            self._copy_counts[("R", v)] = self.kb

        self._original_adj: Dict[Tuple[str, Node], List[OriginalEdge]] = defaultdict(list)
        for u, v in self.g1_edges:
            self._original_adj[("L", u)].append((u, v))
            self._original_adj[("R", v)].append((u, v))

        self._num_vertices = (
            self.k * (len(self.A_left) + len(self.A_right))
            + self.kb * (len(self.B_left) + len(self.B_right))
        )
        self._num_edges = sum(
            self._copy_counts[("L", u)] * self._copy_counts[("R", v)]
            for u, v in self.g1_edges
        )

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

    def vertices(self) -> List[CopiedNode]:
        """
        Return all copied vertices in G1'.
        """
        return [self._copied_vertex_at(i) for i in range(self._num_vertices)]

    def sample_vertices(self, num_samples: int, seed: int | None = None) -> List[CopiedNode]:
        """
        Sample copied vertices uniformly with replacement without materializing all vertices.
        """
        if self._num_vertices == 0 or num_samples <= 0:
            return []

        import random

        rng = random.Random(seed)
        return [
            self._copied_vertex_at(rng.randrange(self._num_vertices))
            for _ in range(num_samples)
        ]

    def incident_edges(self, v: CopiedNode) -> List[CopiedEdge]:
        """
        Return copied edges incident to copied vertex v.
        """
        side, node, copy_index = v
        if side not in {"L", "R"}:
            return []

        own_count = self._copy_count(side, node)
        if copy_index < 0 or copy_index >= own_count:
            return []

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


if __name__ == "__main__":
    # Tiny sanity test using a fake inner oracle.
    class FakeInnerOracle:
        def __init__(self, matched_vertices_set):
            self._matched = set(matched_vertices_set)

        def vertex_matched(self, v):
            return v in self._matched

    U = [0, 1, 2, 3]
    V = [10, 11, 12, 13]
    E = [
        (0, 10),
        (0, 11),
        (1, 10),
        (1, 12),
        (2, 11),
        (2, 13),
        (3, 12),
    ]

    # Explicit M from sparsification
    M = [(0, 10), (1, 12)]

    # Suppose inner oracle says M' matches 2 and 11
    fake_oracle = FakeInnerOracle({2, 11})

    view = Case1CopiedView(
        left_nodes=U,
        right_nodes=V,
        edges=E,
        M=M,
        inner_mprime_oracle=fake_oracle,
        k=2,
    )

    print("A_left  =", view.A_left)
    print("A_right =", view.A_right)
    print("B_left  =", view.B_left)
    print("B_right =", view.B_right)
    print("num copied vertices =", view.num_vertices())
    print("num copied edges    =", view.num_edges())
