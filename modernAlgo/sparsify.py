from __future__ import annotations

import math
import random
from typing import Hashable, List, Set, Tuple

from modernAlgo.graph_oracle import BipartiteGraphOracle

Node = Hashable
Edge = Tuple[Node, Node]


def default_sparsify_sample_count(num_vertices: int) -> int:
    """
    Return the paper's Algorithm 2 sparsification sample count.
    """
    if num_vertices <= 1:
        return 0

    return math.ceil(2 * math.sqrt(num_vertices) * math.log(max(num_vertices, 2)))


def sparsify_partial_matching_from_graph(
    graph: BipartiteGraphOracle,
    seed: int | None = None,
) -> List[Edge]:
    """
    Paper-inspired sparsification step (Algorithm 2).

    Starting from an empty matching M, process vertices one by one.
    For each currently unmatched vertex v, sample

        c = ceil(2 * sqrt(n) * log n)

    random neighbors (with replacement) from N(v). If one sampled neighbor
    is unmatched, add that edge to M and continue.

    Parameters
    ----------
    graph : BipartiteGraphOracle
        Bipartite graph accessed through adjacency-list queries.
    seed : int | None
        Random seed for reproducibility

    Returns
    -------
    List[Edge]
        A partial matching M produced by the sparsification step
    """
    rng = random.Random(seed)

    all_vertices: List[Node] = list(graph.vertices())
    n = len(all_vertices)

    if n <= 1:
        return []

    c = default_sparsify_sample_count(n)

    matched: Set[Node] = set()
    matching: List[Edge] = []

    for v in all_vertices:
        if v in matched:
            continue

        degree = graph.degree(v)
        if degree == 0:
            continue

        for _ in range(c):
            u = graph.neighbor_at(v, rng.randrange(degree))

            if u not in matched and v not in matched:
                # Store edges in canonical bipartite direction: (left, right)
                if graph.side(v) == "L":
                    matching.append((v, u))
                else:
                    matching.append((u, v))

                matched.add(v)
                matched.add(u)
                break

    return matching


