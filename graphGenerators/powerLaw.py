from __future__ import annotations

import random
from typing import List, Tuple, Optional

Node = int
Edge = Tuple[Node, Node]


def generate_powerlaw_bipartite_graph(
    n_left: int,
    n_right: int,
    m: int,
    alpha_left: float = 2.5,
    alpha_right: float = 2.5,
    seed: Optional[int] = None,
) -> Tuple[List[Node], List[Node], List[Edge]]:
    """
    Generate a bipartite graph with power-law-like degree bias using
    weighted endpoint sampling.

    Parameters
    ----------
    n_left : int
        Number of vertices on the left side U.
    n_right : int
        Number of vertices on the right side V.
    m : int
        Target number of edges.
    alpha_left : float
        Power-law exponent controlling left-side skew.
    alpha_right : float
        Power-law exponent controlling right-side skew.
    seed : Optional[int]
        Random seed.

    Returns
    -------
    Tuple[List[Node], List[Node], List[Edge]]
        U, V, edges
    """
    if n_left <= 0 or n_right <= 0:
        raise ValueError("n_left and n_right must be positive")
    if m < 0:
        raise ValueError("m must be nonnegative")
    if alpha_left <= 1.0 or alpha_right <= 1.0:
        raise ValueError("alpha_left and alpha_right must be > 1")

    rng = random.Random(seed)

    U = list(range(n_left))
    V = list(range(n_left, n_left + n_right))

    # Rank-based weights: smaller rank => larger expected degree
    left_weights = [1.0 / ((i + 1) ** (alpha_left - 1.0)) for i in range(n_left)]
    right_weights = [1.0 / ((j + 1) ** (alpha_right - 1.0)) for j in range(n_right)]

    edge_set = set()
    max_edges = n_left * n_right
    target_edges = min(m, max_edges)

    while len(edge_set) < target_edges:
        u = rng.choices(U, weights=left_weights, k=1)[0]
        v = rng.choices(V, weights=right_weights, k=1)[0]
        edge_set.add((u, v))

    edges = list(edge_set)
    rng.shuffle(edges)

    return U, V, edges


def expected_er_edges(n_left: int, n_right: int, p: float) -> int:
    """
    Convenience helper so you can match ER density approximately by setting:
        m ≈ n_left * n_right * p
    """
    return int(round(n_left * n_right * p))

