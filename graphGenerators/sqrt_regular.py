from __future__ import annotations

import math
import random
from typing import List, Tuple, Optional

Node = int
Edge = Tuple[Node, Node]


def generate_near_regular_bipartite_graph(
    n_left: int,
    n_right: int,
    target_degree: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[List[Node], List[Node], List[Edge]]:
    """
    Generate a near-regular bipartite graph with degree about sqrt(n).

    Construction:
    - For each left vertex u, choose `target_degree` distinct right neighbors uniformly.
    - This guarantees exact left degree = target_degree.
    - Right degrees will be concentrated around:
          n_left * target_degree / n_right
      and for balanced graphs (n_left = n_right = n), this is also target_degree.

    Parameters
    ----------
    n_left : int
        Number of left vertices.
    n_right : int
        Number of right vertices.
    target_degree : Optional[int]
        Desired degree on the left side. If None, uses ceil(sqrt(max(n_left, n_right))).
    seed : Optional[int]
        Random seed.

    Returns
    -------
    Tuple[List[Node], List[Node], List[Edge]]
        U, V, edges
    """
    if n_left <= 0 or n_right <= 0:
        raise ValueError("n_left and n_right must be positive")

    rng = random.Random(seed)

    if target_degree is None:
        target_degree = math.ceil(math.sqrt(max(n_left, n_right)))

    if target_degree <= 0:
        raise ValueError("target_degree must be positive")
    if target_degree > n_right:
        raise ValueError("target_degree cannot exceed n_right")

    U = list(range(n_left))
    V = list(range(n_left, n_left + n_right))

    edge_set = set()

    for u in U:
        neighbors = rng.sample(V, target_degree)
        for v in neighbors:
            edge_set.add((u, v))

    edges = list(edge_set)
    rng.shuffle(edges)

    return U, V, edges


if __name__ == "__main__":
    U, V, edges = generate_near_regular_bipartite_graph(
        n_left=100,
        n_right=100,
        target_degree=None,   # defaults to ceil(sqrt(100)) = 10
        seed=42,
    )

    print("Left vertices:", len(U))
    print("Right vertices:", len(V))
    print("Edges:", len(edges))
    print("Expected left degree:", math.ceil(math.sqrt(100)))
    print("Average degree on left:", len(edges) / len(U))