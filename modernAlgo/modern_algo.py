from __future__ import annotations

import math
import time
from typing import Any, Dict, Hashable, Iterable, Sequence, Tuple

from modernAlgo.case1.estimator import run_case1_oracle
from modernAlgo.case2.estimator import run_case2_oracle
from modernAlgo.graph_oracle import BipartiteGraphOracle, EdgeListBipartiteGraph
from modernAlgo.sparsify import (
    default_sparsify_sample_count,
    sparsify_partial_matching_from_graph,
)

Node = Hashable
Edge = Tuple[Node, Node]
DEFAULT_PAPER_EPSILON = 0.5


def theorem_k_from_epsilon(epsilon: float) -> int:
    """
    Return the paper-style capacity parameter k for a target epsilon.
    """
    if not (0.0 < epsilon < 1.0):
        raise ValueError("epsilon must be in (0, 1)")

    b = 1.0 + math.sqrt(2.0)
    return math.floor(1.0 / (b * (epsilon ** 3))) + 1


def choose_capacity_parameter(
    k: int | None = None,
    epsilon: float | None = None,
) -> tuple[int, str]:
    """
    Choose the effective copied-graph capacity parameter.

    Explicit k takes precedence. If k is omitted, use the paper-style lower
    bound k > 1 / (b * epsilon^3), with DEFAULT_PAPER_EPSILON when epsilon is
    omitted.
    """
    effective_epsilon = DEFAULT_PAPER_EPSILON if epsilon is None else epsilon
    if not (0.0 < effective_epsilon < 1.0):
        raise ValueError("epsilon must be in (0, 1)")

    if k is not None:
        if k <= 0:
            raise ValueError("k must be positive")
        return k, "explicit"

    if epsilon is None:
        return theorem_k_from_epsilon(effective_epsilon), "default_epsilon"

    return theorem_k_from_epsilon(effective_epsilon), "epsilon"


def run_modern_oracle(
    left_nodes: Sequence[Node],
    right_nodes: Sequence[Node],
    edges: Iterable[Edge],
    k: int | None = None,
    epsilon: float | None = DEFAULT_PAPER_EPSILON,
    seed: int | None = None,
) -> Dict[str, Any]:
    """
    Full bipartite oracle-based implementation of the modern algorithm.

    Pipeline:
    1. Build explicit sparsification matching M
    2. Run Case 1 oracle estimator to get mu1
    3. Run Case 2 oracle estimator to get mu2
    4. Return max(mu1, mu2)

    Parameters
    ----------
    left_nodes : Sequence[Node]
        Left partition U
    right_nodes : Sequence[Node]
        Right partition V
    edges : Iterable[Edge]
        Bipartite edge list
    k : int | None
        Capacity parameter used in copied b-matching graphs. If None and
        epsilon is provided, use the paper-style k selection.
    epsilon : float | None
        Optional theorem parameter used to choose k > 1 / (b * epsilon^3)
        when k is not explicitly provided. Defaults to DEFAULT_PAPER_EPSILON.
    seed : int | None
        Random seed for reproducibility

    Returns
    -------
    Dict[str, Any]
        Dictionary containing the final estimate and estimator metadata.
    """
    graph = EdgeListBipartiteGraph(left_nodes, right_nodes, edges)
    return run_modern_graph_oracle(graph, k=k, epsilon=epsilon, seed=seed)


def run_modern_graph_oracle(
    graph: BipartiteGraphOracle,
    k: int | None = None,
    epsilon: float | None = DEFAULT_PAPER_EPSILON,
    seed: int | None = None,
) -> Dict[str, Any]:
    """
    Full bipartite oracle-based implementation of the modern algorithm.

    This is the graph-oracle entry point. It assumes adjacency-list access
    through degree(v) and neighbor_at(v, i). The edge-list entry point above
    exists only to adapt the repository's experiment generators.
    """
    effective_k, k_source = choose_capacity_parameter(k=k, epsilon=epsilon)

    # Step 1: explicit sparsification matching M
    start = time.perf_counter()
    M = sparsify_partial_matching_from_graph(
        graph=graph,
        seed=seed,
    )
    t_sparsify = time.perf_counter() - start

    # Step 2: Case 1
    start = time.perf_counter()
    case1_result = run_case1_oracle(
        M=M,
        k=effective_k,
        seed=seed,
        graph=graph,
    )
    t_case1 = time.perf_counter() - start

    # Step 3: Case 2
    start = time.perf_counter()
    case2_result = run_case2_oracle(
        M=M,
        k=effective_k,
        seed=seed,
        graph=graph,
    )
    t_case2 = time.perf_counter() - start

    mu1 = case1_result["mu1"]
    mu2 = case2_result["mu2"]
    final_estimate = max(mu1, mu2)
    num_vertices = graph.num_vertices()
    effective_sparsify_c = default_sparsify_sample_count(num_vertices)

    return {
        "estimate": final_estimate,
        "mu1": mu1,
        "mu2": mu2,
        "M": M,
        "M_size": len(M),
        "k": effective_k,
        "k_source": k_source,
        "epsilon": epsilon,
        "sparsify_c": effective_sparsify_c,
        "t_sparsify": t_sparsify,
        "t_case1": t_case1,
        "t_case2": t_case2,
        "case1": case1_result,
        "case2": case2_result,
    }
