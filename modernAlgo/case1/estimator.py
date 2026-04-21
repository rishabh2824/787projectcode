from __future__ import annotations

import math
from typing import Hashable, Iterable, Sequence, Tuple, Dict, Any

from modernAlgo.behnezhad_rgmm_oracle import BehnezhadRGMMOracle
from modernAlgo.case1.view_copied import Case1CopiedView
from modernAlgo.case1.view_unmatched import UnmatchedInducedView

Node = Hashable
Edge = Tuple[Node, Node]


def paper_additive_slack(num_original_vertices: int) -> float:
    """
    Return the paper's n / (2 log n) additive estimate slack.
    """
    return num_original_vertices / (2.0 * math.log(max(num_original_vertices, 2)))


def paper_sample_count(num_original_vertices: int) -> int:
    """
    Return the paper's r = 6 log^3 n estimator sample count.
    """
    return max(1, math.ceil(6 * (math.log(max(num_original_vertices, 2)) ** 3)))


def apply_additive_slack(estimate: float, num_original_vertices: int) -> float:
    """
    Bias an estimate downward by the paper's additive slack.
    """
    return max(0.0, estimate - paper_additive_slack(num_original_vertices))


def estimate_mprime_size_from_inner_oracle(
    left_nodes: Sequence[Node],
    right_nodes: Sequence[Node],
    edges: Iterable[Edge],
    M: Iterable[Edge],
    seed: int | None = None,
) -> Dict[str, Any]:
    """
    Estimate |M'| using the inner RGMM oracle on G[V \\ V(M)].

    Returns a dictionary containing:
    - Mprime_estimate
    - matched_fraction
    - num_samples
    - unmatched_num_vertices
    - unmatched_num_edges
    - inner_oracle
    - unmatched_view
    """
    view = UnmatchedInducedView(
        left_nodes=left_nodes,
        right_nodes=right_nodes,
        edges=edges,
        M=M,
    )

    unmatched_num_vertices = view.num_vertices()
    n_total = len(left_nodes) + len(right_nodes)

    if unmatched_num_vertices == 0:
        return {
            "Mprime_estimate": 0.0,
            "matched_fraction": 0.0,
            "num_samples": 0,
            "additive_slack": paper_additive_slack(n_total),
            "unmatched_num_vertices": 0,
            "unmatched_num_edges": view.num_edges(),
            "inner_oracle": None,
            "unmatched_view": view,
        }

    num_samples = paper_sample_count(n_total)

    inner_oracle = BehnezhadRGMMOracle(
        view,
        seed=None if seed is None else seed + 5_000_003,
    )

    sampled_vertices = view.sample_vertices(
        num_samples=num_samples,
        seed=seed,
    )

    matched_fraction = inner_oracle.matched_vertices_fraction(sampled_vertices)

    # Same logic: if fraction f of vertices are matched, matching size is f * |V| / 2
    Mprime_estimate = apply_additive_slack(
        matched_fraction * unmatched_num_vertices / 2.0,
        n_total,
    )

    return {
        "Mprime_estimate": Mprime_estimate,
        "matched_fraction": matched_fraction,
        "num_samples": num_samples,
        "additive_slack": paper_additive_slack(n_total),
        "unmatched_num_vertices": unmatched_num_vertices,
        "unmatched_num_edges": view.num_edges(),
        "inner_oracle": inner_oracle,
        "unmatched_view": view,
    }


def estimate_b1_size_from_outer_oracle(
    left_nodes: Sequence[Node],
    right_nodes: Sequence[Node],
    edges: Iterable[Edge],
    M: Iterable[Edge],
    inner_oracle: BehnezhadRGMMOracle,
    k: int,
    seed: int | None = None,
) -> Dict[str, Any]:
    """
    Estimate |B1| using the outer RGMM oracle on the copied Case 1 graph G1'.

    Returns a dictionary containing:
    - B1_estimate
    - matched_fraction
    - num_samples
    - copied_num_vertices
    - copied_num_edges
    - outer_oracle
    - copied_view
    """
    if k <= 0:
        raise ValueError("k must be positive")

    b = 1.0 + math.sqrt(2.0)

    copied_view = Case1CopiedView(
        left_nodes=left_nodes,
        right_nodes=right_nodes,
        edges=edges,
        M=M,
        inner_mprime_oracle=inner_oracle,
        k=k,
        b=b,
    )

    copied_num_vertices = copied_view.num_vertices()
    n_total = len(left_nodes) + len(right_nodes)

    if copied_num_vertices == 0:
        return {
            "B1_estimate": 0.0,
            "matched_fraction": 0.0,
            "num_samples": 0,
            "additive_slack": paper_additive_slack(n_total),
            "copied_num_vertices": 0,
            "copied_num_edges": copied_view.num_edges(),
            "outer_oracle": None,
            "copied_view": copied_view,
        }

    num_samples = paper_sample_count(n_total)

    # Important: outer oracle needs its own independent random ordering.
    outer_oracle = BehnezhadRGMMOracle(
        copied_view,
        seed=None if seed is None else seed + 30_000_003,
    )

    sampled_vertices = copied_view.sample_vertices(
        num_samples=num_samples,
        seed=None if seed is None else seed + 20_000_003,
    )

    matched_fraction = outer_oracle.matched_vertices_fraction(sampled_vertices)
    B1_estimate = apply_additive_slack(
        matched_fraction * copied_num_vertices / 2.0,
        n_total,
    )

    return {
        "B1_estimate": B1_estimate,
        "matched_fraction": matched_fraction,
        "num_samples": num_samples,
        "additive_slack": paper_additive_slack(n_total),
        "copied_num_vertices": copied_num_vertices,
        "copied_num_edges": copied_view.num_edges(),
        "outer_oracle": outer_oracle,
        "copied_view": copied_view,
    }


def compute_mu1_from_estimates(
    M: Iterable[Edge],
    Mprime_estimate: float,
    B1_estimate: float,
    k: int,
) -> float:
    """
    Compute the Case 1 estimate:

        mu1 = |M| + (1 - 1/b) * mu_M' + (1/(k*b)) * mu_B1
    """
    if k <= 0:
        raise ValueError("k must be positive")

    b = 1.0 + math.sqrt(2.0)
    M_size = len(list(M))

    return M_size + (1.0 - 1.0 / b) * Mprime_estimate + (1.0 / (k * b)) * B1_estimate


def run_case1_oracle(
    left_nodes: Sequence[Node],
    right_nodes: Sequence[Node],
    edges: Iterable[Edge],
    M: Iterable[Edge],
    k: int = 10,
    seed: int | None = None,
) -> Dict[str, Any]:
    """
    Full oracle-based Case 1 pipeline.

    Steps:
    1. Build unmatched-induced view G[V \\ V(M)]
    2. Create inner RGMM oracle and estimate |M'|
    3. Build copied Case 1 graph G1' using the inner oracle
    4. Create outer RGMM oracle and estimate |B1|
    5. Compute mu1

    Returns a dictionary containing:
    - mu1
    - Mprime_estimate
    - B1_estimate
    - inner_matched_fraction
    - outer_matched_fraction
    - inner_num_samples
    - outer_num_samples
    - unmatched_num_vertices
    - unmatched_num_edges
    - copied_num_vertices
    - copied_num_edges
    - inner_oracle
    - outer_oracle
    - unmatched_view
    - copied_view
    """
    est_mprime = estimate_mprime_size_from_inner_oracle(
        left_nodes=left_nodes,
        right_nodes=right_nodes,
        edges=edges,
        M=M,
        seed=seed,
    )

    inner_oracle = est_mprime["inner_oracle"]

    est_b1 = estimate_b1_size_from_outer_oracle(
        left_nodes=left_nodes,
        right_nodes=right_nodes,
        edges=edges,
        M=M,
        inner_oracle=inner_oracle,
        k=k,
        seed=seed,
    )

    mu1 = compute_mu1_from_estimates(
        M=M,
        Mprime_estimate=est_mprime["Mprime_estimate"],
        B1_estimate=est_b1["B1_estimate"],
        k=k,
    )

    return {
        "mu1": mu1,
        "Mprime_estimate": est_mprime["Mprime_estimate"],
        "B1_estimate": est_b1["B1_estimate"],
        "Mprime_additive_slack": est_mprime["additive_slack"],
        "B1_additive_slack": est_b1["additive_slack"],
        "inner_matched_fraction": est_mprime["matched_fraction"],
        "outer_matched_fraction": est_b1["matched_fraction"],
        "inner_num_samples": est_mprime["num_samples"],
        "outer_num_samples": est_b1["num_samples"],
        "unmatched_num_vertices": est_mprime["unmatched_num_vertices"],
        "unmatched_num_edges": est_mprime["unmatched_num_edges"],
        "copied_num_vertices": est_b1["copied_num_vertices"],
        "copied_num_edges": est_b1["copied_num_edges"],
        "inner_oracle": est_mprime["inner_oracle"],
        "outer_oracle": est_b1["outer_oracle"],
        "unmatched_view": est_mprime["unmatched_view"],
        "copied_view": est_b1["copied_view"],
    }
