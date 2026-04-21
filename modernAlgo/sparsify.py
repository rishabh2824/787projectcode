from __future__ import annotations
import math
import random
from collections import defaultdict
from typing import Dict, Hashable, Iterable, List, Sequence, Set, Tuple

from modernAlgo.graph_oracle import BipartiteGraphOracle, EdgeListBipartiteGraph

Node = Hashable
Edge = Tuple[Node, Node]


def default_sparsify_sample_count(num_vertices: int) -> int:
    """
    Return the paper's Algorithm 2 sparsification sample count.
    """
    if num_vertices <= 1:
        return 0

    return math.ceil(2 * math.sqrt(num_vertices) * math.log(max(num_vertices, 2)))


def build_adjacency_from_edges(
    left_nodes: Sequence[Node],
    right_nodes: Sequence[Node],
    edges: Iterable[Edge],
) -> Dict[Node, List[Node]]:
    """
    Build an undirected adjacency list from a bipartite edge list.

    Returns
    -------
    Dict[Node, List[Node]]
        adjacency list for all vertices in U union V
    """
    adjacency: Dict[Node, List[Node]] = defaultdict(list)

    for u in left_nodes:
        adjacency[u] = []
    for v in right_nodes:
        adjacency[v] = []

    for u, v in edges:
        adjacency[u].append(v)
        adjacency[v].append(u)

    return adjacency


def sparsify_partial_matching(
    left_nodes: Sequence[Node],
    right_nodes: Sequence[Node],
    edges: Iterable[Edge],
    seed: int | None = None,
) -> List[Edge]:
    graph = EdgeListBipartiteGraph(left_nodes, right_nodes, edges)
    return sparsify_partial_matching_from_graph(graph, seed=seed)


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


