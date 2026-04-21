from __future__ import annotations

from typing import Hashable, Iterable, List, Sequence, Set, Tuple

Node = Hashable
Edge = Tuple[Node, Node]


def greedy_bipartite_matching(
    left_nodes: Sequence[Node],
    right_nodes: Sequence[Node],
    edges: Iterable[Edge],
) -> List[Edge]:
    """
    Return a greedy maximal matching for a bipartite graph.
    """
    matched_u: Set[Node] = set()
    matched_v: Set[Node] = set()
    matching: List[Edge] = []

    for u, v in edges:
        if u not in matched_u and v not in matched_v:
            matching.append((u, v))
            matched_u.add(u)
            matched_v.add(v)

    return matching
