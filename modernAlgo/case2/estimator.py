from __future__ import annotations

import math
from typing import Hashable, Iterable, Sequence, Tuple, Dict, Any

from modernAlgo.ranks import LazyEdgeRanks
from modernAlgo.random_neighbor_rgmm_oracle import RandomNeighborRGMMOracle
from modernAlgo.case2.copied_graph import Case2CopiedView
from modernAlgo.case1.estimator import (
    apply_additive_slack,
    paper_additive_slack,
    paper_sample_count,
)

Node = Hashable
Edge = Tuple[Node, Node]


def estimate_b2_size_from_oracle(
    left_nodes: Sequence[Node],
    right_nodes: Sequence[Node],
    edges: Iterable[Edge],
    M: Iterable[Edge],
    k: int,
    seed: int | None = None,
) -> Dict[str, Any]:
    """
    Estimate |B2| using the RGMM oracle on the copied Case 2 graph G2'.

    This mirrors the role of Algorithm 5 in the paper. The copied graph view
    is lazy, and copied vertices are sampled without materializing all copied
    vertices or copied edges.

    Returns a dictionary with:
    - B2_estimate
    - matched_fraction
    - num_samples
    - copied_num_vertices
    - copied_num_edges
    - oracle
    - view
    """
    if k <= 0:
        raise ValueError("k must be positive")

    b = 1.0 + math.sqrt(2.0)

    view = Case2CopiedView(
        left_nodes=left_nodes,
        right_nodes=right_nodes,
        edges=edges,
        M=M,
        k=k,
        b=b,
    )

    copied_num_vertices = view.num_vertices()
    n_total = len(left_nodes) + len(right_nodes)

    if copied_num_vertices == 0:
        return {
            "B2_estimate": 0.0,
            "matched_fraction": 0.0,
            "num_samples": 0,
            "additive_slack": paper_additive_slack(n_total),
            "copied_num_vertices": 0,
            "copied_num_edges": view.num_edges(),
            "oracle": None,
            "view": view,
        }

    num_samples = paper_sample_count(n_total)

    ranks = LazyEdgeRanks(seed=seed)
    oracle = RandomNeighborRGMMOracle(
        view,
        ranks,
        seed=None if seed is None else seed + 10_000_003,
    )

    sampled_vertices = view.sample_vertices(
        num_samples=num_samples,
        seed=seed,
    )

    matched_fraction = oracle.matched_vertices_fraction(sampled_vertices)

    # If a fraction f of copied vertices are matched, estimated matching size is:
    #   f * |V(G2')| / 2
    # since each matching edge matches exactly 2 copied vertices.
    B2_estimate = apply_additive_slack(
        matched_fraction * copied_num_vertices / 2.0,
        n_total,
    )

    return {
        "B2_estimate": B2_estimate,
        "matched_fraction": matched_fraction,
        "num_samples": num_samples,
        "additive_slack": paper_additive_slack(n_total),
        "copied_num_vertices": copied_num_vertices,
        "copied_num_edges": view.num_edges(),
        "oracle": oracle,
        "view": view,
    }


def compute_mu2_from_b2_estimate(
    M: Iterable[Edge],
    B2_estimate: float,
    k: int,
) -> float:
    """
    Compute the Case 2 estimate:

        mu2 = (1 - 1/b) * |M| + (1 / (k*b)) * mu_B2

    where b = 1 + sqrt(2).
    """
    if k <= 0:
        raise ValueError("k must be positive")

    b = 1.0 + math.sqrt(2.0)
    M_size = len(list(M))

    return (1.0 - 1.0 / b) * M_size + (1.0 / (k * b)) * B2_estimate


def run_case2_oracle(
    left_nodes: Sequence[Node],
    right_nodes: Sequence[Node],
    edges: Iterable[Edge],
    M: Iterable[Edge],
    k: int = 10,
    seed: int | None = None,
) -> Dict[str, Any]:
    """
    Full oracle-based Case 2 pipeline.

    Returns a dictionary containing:
    - mu2
    - B2_estimate
    - matched_fraction
    - num_samples
    - copied_num_vertices
    - copied_num_edges
    - oracle
    - view
    """
    est = estimate_b2_size_from_oracle(
        left_nodes=left_nodes,
        right_nodes=right_nodes,
        edges=edges,
        M=M,
        k=k,
        seed=seed,
    )

    mu2 = compute_mu2_from_b2_estimate(
        M=M,
        B2_estimate=est["B2_estimate"],
        k=k,
    )

    return {
        "mu2": mu2,
        "B2_estimate": est["B2_estimate"],
        "B2_additive_slack": est["additive_slack"],
        "matched_fraction": est["matched_fraction"],
        "num_samples": est["num_samples"],
        "copied_num_vertices": est["copied_num_vertices"],
        "copied_num_edges": est["copied_num_edges"],
        "oracle": est["oracle"],
        "view": est["view"],
    }
