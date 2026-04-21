from __future__ import annotations

from typing import List, Tuple, Optional
import random

Node = int
Edge = Tuple[Node, Node]


def generate_er_bipartite_graph(
    n_left: int,
    n_right: int,
    p: float,
    seed: Optional[int] = None,
) -> Tuple[List[Node], List[Node], List[Edge]]:
    """
    Generate an Erdős–Rényi bipartite graph G = (U, V, E).

    Parameters
    ----------
    n_left : int
        Number of vertices in the left partition U.
    n_right : int
        Number of vertices in the right partition V.
    p : float
        Probability of including each possible edge (u, v),
        where u in U and v in V.
    seed : Optional[int]
        Random seed for reproducibility.

    Returns
    -------
    Tuple[List[Node], List[Node], List[Edge]]
        U     : list of left-side vertices
        V     : list of right-side vertices
        edges : list of edges (u, v) with u in U and v in V
    """
    if n_left < 0 or n_right < 0:
        raise ValueError("n_left and n_right must be nonnegative")
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be between 0 and 1")

    rng = random.Random(seed)

    # Left partition: 0, 1, ..., n_left - 1
    U = list(range(n_left))

    # Right partition: n_left, n_left + 1, ..., n_left + n_right - 1
    # This ensures U and V are disjoint.
    V = list(range(n_left, n_left + n_right))

    edges: List[Edge] = []

    for u in U:
        for v in V:
            if rng.random() < p:
                edges.append((u, v))

    return U, V, edges

