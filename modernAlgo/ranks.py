from __future__ import annotations

from typing import Hashable, Tuple

Node = Hashable
Edge = Tuple[Node, Node]


def canonical_edge(edge: Edge) -> Edge:
    """
    Return a canonical representation of an undirected edge.

    Example:
        (5, 2) -> (2, 5)

    This ensures that the same undirected edge always gets the same key.
    """
    u, v = edge
    return (u, v) if u <= v else (v, u)
