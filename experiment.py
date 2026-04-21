from __future__ import annotations

import math
import argparse
import time
from typing import List

from graphGenerators.ERgraph import generate_er_bipartite_graph
from graphGenerators.powerLaw import (generate_powerlaw_bipartite_graph, expected_er_edges)
from graphGenerators.sqrt_regular import generate_near_regular_bipartite_graph
from greedy import greedy_bipartite_matching
from hopcroft import maximum_matching_size_networkx
from modernAlgo.modern_algo import (
    DEFAULT_PAPER_EPSILON,
    run_modern_oracle,
    theorem_k_from_epsilon,
)


def generate_graph(
    graph_type: str,
    n: int,
    param: float,
    seed: int,
    alpha_left: float = 2.5,
    alpha_right: float = 2.5,
):
    """
    graph_type:
        - "er":         param = p
        - "powerlaw":   param = p-like density control, converted to m = n^2 * param
        - "sqrtreg":    ignores param, uses target_degree = ceil(sqrt(n))
    """
    if graph_type == "er":
        U, V, edges = generate_er_bipartite_graph(n, n, param, seed=seed)

    elif graph_type == "powerlaw":
        m = expected_er_edges(n, n, param)
        U, V, edges = generate_powerlaw_bipartite_graph(
            n_left=n,
            n_right=n,
            m=m,
            alpha_left=alpha_left,
            alpha_right=alpha_right,
            seed=seed,
        )

    elif graph_type == "sqrtreg":
        target_degree = math.ceil(math.sqrt(n))
        U, V, edges = generate_near_regular_bipartite_graph(
            n_left=n,
            n_right=n,
            target_degree=target_degree,
            seed=seed,
        )

    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")

    return U, V, edges


def run_single_experiment(
    graph_type: str,
    n: int,
    param: float,
    seed: int,
    k: int | None = None,
    epsilon: float = DEFAULT_PAPER_EPSILON,
    alpha_left: float = 2.5,
    alpha_right: float = 2.5,
) -> None:
    U, V, edges = generate_graph(
        graph_type=graph_type,
        n=n,
        param=param,
        seed=seed,
        alpha_left=alpha_left,
        alpha_right=alpha_right,
    )

    # Greedy
    t0 = time.perf_counter()
    greedy_matching = greedy_bipartite_matching(U, V, edges)
    t1 = time.perf_counter()

    greedy_size = len(greedy_matching)
    greedy_time = t1 - t0

    # Modern oracle algorithm
    t2 = time.perf_counter()
    modern_result = run_modern_oracle(
        U,
        V,
        edges,
        k=k,
        epsilon=epsilon,
        seed=seed,
    )
    t3 = time.perf_counter()

    modern_estimate = modern_result["estimate"]
    mu1 = modern_result["mu1"]
    mu2 = modern_result["mu2"]
    M_size = modern_result["M_size"]
    effective_k = modern_result["k"]
    k_source = modern_result["k_source"]
    effective_sparsify_c = modern_result["sparsify_c"]
    additive_slack = modern_result["case1"]["Mprime_additive_slack"]

    case1 = modern_result["case1"]
    case2 = modern_result["case2"]

    unmatched_num_vertices = case1["unmatched_num_vertices"]
    unmatched_num_edges = case1["unmatched_num_edges"]

    M_prime_estimate = case1["Mprime_estimate"]
    B1_estimate = case1["B1_estimate"]
    B2_estimate = case2["B2_estimate"]

    modern_time = t3 - t2
    selected_case = "case1" if mu1 >= mu2 else "case2"

    # Hopcroft–Karp
    t4 = time.perf_counter()
    optimal_size = maximum_matching_size_networkx(U, V, edges)
    t5 = time.perf_counter()

    hk_time = t5 - t4

    if optimal_size == 0:
        greedy_ratio = 1.0
        modern_ratio = 1.0
    else:
        greedy_ratio = greedy_size / optimal_size
        modern_ratio = modern_estimate / optimal_size

    print(
        f"type={graph_type:8s}, n={n:4d}, param={param:.3f}, edges={len(edges):6d}, "
        f"greedy={greedy_size:4d}, modern={modern_estimate:8.4f}, optimal={optimal_size:4d}, "
        f"greedy_ratio={greedy_ratio:.4f}, modern_ratio={modern_ratio:.4f}, "
        f"selected={selected_case:5s}, k={effective_k:4d}({k_source}), "
        f"sparsify_c={effective_sparsify_c:4d}, "
        f"add_slack={additive_slack:8.4f}, "
        f"|M|={M_size:4d}, unmatched_V={unmatched_num_vertices:4d}, unmatched_E={unmatched_num_edges:4d}, "
        f"M'_est={M_prime_estimate:8.4f}, "
        f"B1_est={B1_estimate:8.4f}, B2_est={B2_estimate:8.4f}, "
        f"mu1={mu1:.4f}, mu2={mu2:.4f}, "
        f"t_greedy={greedy_time:.6f}s, "
        f"t_modern={modern_time:.6f}s, "
        f"t_hk={hk_time:.6f}s"
    )


def run_experiments(
    graph_type: str,
    n_values: List[int],
    param_values: List[float],
    num_trials: int = 3,
    k: int | None = None,
    epsilon: float = DEFAULT_PAPER_EPSILON,
    alpha_left: float = 2.5,
    alpha_right: float = 2.5,
) -> None:
    for n in n_values:
        for param in param_values:
            for trial in range(num_trials):
                seed = 1000 * n + 100 * trial + int(1000 * param)

                run_single_experiment(
                    graph_type=graph_type,
                    n=n,
                    param=param,
                    seed=seed,
                    k=k,
                    epsilon=epsilon,
                    alpha_left=alpha_left,
                    alpha_right=alpha_right,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_PAPER_EPSILON,
        help=(
            "Theorem epsilon used to choose k > 1 / ((1+sqrt(2))*epsilon^3). "
            f"Default: {DEFAULT_PAPER_EPSILON}."
        ),
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Explicit copied-graph capacity parameter. Overrides epsilon-derived k.",
    )
    args = parser.parse_args()

    if args.epsilon is not None and not (0.0 < args.epsilon < 1.0):
        parser.error("--epsilon must be in (0, 1)")

    if args.k is not None and args.k <= 0:
        parser.error("--k must be positive")

    if args.k is None:
        derived_k = theorem_k_from_epsilon(args.epsilon)
        print(f"=== Paper-aligned mode: epsilon={args.epsilon}, derived k={derived_k} ===")
    else:
        print(f"=== Paper-aligned mode: explicit k={args.k}, epsilon={args.epsilon} ===")

    n_values = [50, 50, 100]
    param_values = [0.01, 0.05, 0.1]

    print("=== ER graphs ===")
    run_experiments(
        graph_type="er",
        n_values=n_values,
        param_values=param_values,
        num_trials=3,
        k=args.k,
        epsilon=args.epsilon,
    )

    print("\n=== Power-law bipartite graphs ===")
    run_experiments(
        graph_type="powerlaw",
        n_values=n_values,
        param_values=param_values,
        num_trials=3,
        k=args.k,
        epsilon=args.epsilon,
        alpha_left=2.5,
        alpha_right=2.5,
    )

    print("\n=== Near-regular sqrt(n)-degree bipartite graphs ===")
    # param is unused for sqrtreg, but we keep the same driver interface
    run_experiments(
        graph_type="sqrtreg",
        n_values=n_values,
        param_values=[0.0],
        num_trials=3,
        k=args.k,
        epsilon=args.epsilon,
    )
