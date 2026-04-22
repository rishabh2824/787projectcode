from __future__ import annotations

import argparse
import csv
import math
import random
import time
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple

from graphGenerators.ERgraph import generate_er_bipartite_graph
from graphGenerators.powerLaw import expected_er_edges, generate_powerlaw_bipartite_graph
from graphGenerators.sqrt_regular import generate_near_regular_bipartite_graph
from greedy import greedy_bipartite_matching
from hopcroft import maximum_matching_size_networkx
from modernAlgo.modern_algo import DEFAULT_PAPER_EPSILON, run_modern_oracle

Node = int
Edge = Tuple[Node, Node]

DEFAULT_TIER1_N_VALUES = "50,100,250,500,1000"
DEFAULT_TIER1_DEGREES = "2,4,8,16,32"
DEFAULT_TIER1_TRIALS = 20
DEFAULT_TIER1_EPSILON = 0.3
DEFAULT_TIER2_N_VALUES = "50,100,250"
DEFAULT_TIER2_DEGREES = "4,8,16"
DEFAULT_TIER2_EPSILONS = "0.7,0.6,0.5,0.4,0.3,0.2,0.1"
DEFAULT_TIER2_TRIALS = 10
DEFAULT_OUTPUT = "results/raw_trials.csv"


CSV_FIELDS = [
    "graph_type",
    "n_left",
    "n_right",
    "num_vertices",
    "num_edges",
    "density",
    "expected_average_degree",
    "average_degree",
    "max_degree",
    "degree_variance",
    "degree_gini",
    "param",
    "trial",
    "seed",
    "optimal_size",
    "greedy_size",
    "random_greedy_size",
    "modern_estimate",
    "modern_mu1",
    "modern_mu2",
    "selected_case",
    "M_size",
    "Mprime_estimate",
    "B1_estimate",
    "B2_estimate",
    "unmatched_num_vertices",
    "copied_num_vertices",
    "case1_copied_num_vertices",
    "case2_copied_num_vertices",
    "epsilon",
    "k",
    "sparsify_c",
    "greedy_ratio",
    "random_greedy_ratio",
    "modern_ratio",
    "modern_error",
    "modern_relative_error",
    "t_greedy",
    "t_random_greedy",
    "t_modern",
    "t_modern_sparsify",
    "t_modern_case1",
    "t_modern_case2",
    "t_modern_other",
    "t_hopcroft",
]


def parse_number_list(raw: str, cast):
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def generate_graph(
    graph_type: str,
    n: int,
    param: float,
    seed: int,
    alpha_left: float,
    alpha_right: float,
) -> tuple[List[Node], List[Node], List[Edge]]:
    if graph_type == "er":
        return generate_er_bipartite_graph(n, n, param, seed=seed)

    if graph_type == "powerlaw":
        m = expected_er_edges(n, n, param)
        return generate_powerlaw_bipartite_graph(
            n_left=n,
            n_right=n,
            m=m,
            alpha_left=alpha_left,
            alpha_right=alpha_right,
            seed=seed,
        )

    if graph_type == "sqrtreg":
        target_degree = math.ceil(math.sqrt(n))
        return generate_near_regular_bipartite_graph(
            n_left=n,
            n_right=n,
            target_degree=target_degree,
            seed=seed,
        )

    raise ValueError(f"unknown graph_type: {graph_type}")


def density_from_expected_degree(expected_degree: float, n: int) -> float:
    if n <= 0:
        raise ValueError("n must be positive")

    return min(1.0, max(0.0, expected_degree / n))


def reported_expected_degree(graph_type: str, n: int, expected_degree: float) -> float:
    if graph_type == "sqrtreg":
        return float(math.ceil(math.sqrt(n)))

    return expected_degree


def degree_gini(degrees: Sequence[int]) -> float:
    if not degrees:
        return 0.0

    total = sum(degrees)
    if total == 0:
        return 0.0

    sorted_degrees = sorted(degrees)
    n = len(sorted_degrees)
    weighted_sum = sum((index + 1) * degree for index, degree in enumerate(sorted_degrees))
    return (2.0 * weighted_sum) / (n * total) - (n + 1.0) / n


def graph_metrics(
    left_nodes: Sequence[Node],
    right_nodes: Sequence[Node],
    edges: Iterable[Edge],
) -> Dict[str, float | int]:
    edge_list = list(edges)
    degrees = {v: 0 for v in list(left_nodes) + list(right_nodes)}
    for u, v in edge_list:
        degrees[u] += 1
        degrees[v] += 1

    n_left = len(left_nodes)
    n_right = len(right_nodes)
    num_vertices = n_left + n_right
    num_edges = len(edge_list)
    possible_edges = n_left * n_right
    degree_values = list(degrees.values())
    avg_degree = (2.0 * num_edges / num_vertices) if num_vertices else 0.0
    degree_variance = (
        mean((degree - avg_degree) ** 2 for degree in degree_values)
        if degree_values
        else 0.0
    )

    return {
        "n_left": n_left,
        "n_right": n_right,
        "num_vertices": num_vertices,
        "num_edges": num_edges,
        "density": (num_edges / possible_edges) if possible_edges else 0.0,
        "average_degree": avg_degree,
        "max_degree": max(degree_values, default=0),
        "degree_variance": degree_variance,
        "degree_gini": degree_gini(degree_values),
    }


def timed_greedy(
    left_nodes: Sequence[Node],
    right_nodes: Sequence[Node],
    edges: Sequence[Edge],
) -> tuple[int, float]:
    start = time.perf_counter()
    matching = greedy_bipartite_matching(left_nodes, right_nodes, edges)
    elapsed = time.perf_counter() - start
    return len(matching), elapsed


def timed_random_greedy(
    left_nodes: Sequence[Node],
    right_nodes: Sequence[Node],
    edges: Sequence[Edge],
    seed: int,
) -> tuple[int, float]:
    shuffled_edges = list(edges)
    random.Random(seed).shuffle(shuffled_edges)
    return timed_greedy(left_nodes, right_nodes, shuffled_edges)


def ratio(value: float, optimum: int) -> float:
    if optimum == 0:
        return 1.0
    return value / optimum


def relative_error(value: float, optimum: int) -> float:
    if optimum == 0:
        return 0.0
    return (value - optimum) / optimum


def run_trial(
    graph_type: str,
    n: int,
    expected_degree: float,
    trial: int,
    epsilon: float,
    k: int | None,
    alpha_left: float,
    alpha_right: float,
    base_seed: int,
) -> Dict[str, float | int | str]:
    param = density_from_expected_degree(expected_degree, n)
    seed = (
        base_seed
        + 1_000_000 * n
        + 10_000 * trial
        + int(round(10_000 * expected_degree))
        + int(round(1_000_000 * epsilon))
    )
    left_nodes, right_nodes, edges = generate_graph(
        graph_type=graph_type,
        n=n,
        param=param,
        seed=seed,
        alpha_left=alpha_left,
        alpha_right=alpha_right,
    )

    metrics = graph_metrics(left_nodes, right_nodes, edges)

    greedy_size, t_greedy = timed_greedy(left_nodes, right_nodes, edges)
    random_greedy_size, t_random_greedy = timed_random_greedy(
        left_nodes,
        right_nodes,
        edges,
        seed=seed + 101,
    )

    start = time.perf_counter()
    modern_result = run_modern_oracle(
        left_nodes,
        right_nodes,
        edges,
        epsilon=epsilon,
        k=k,
        seed=seed,
    )
    t_modern = time.perf_counter() - start
    t_modern_sparsify = modern_result["t_sparsify"]
    t_modern_case1 = modern_result["t_case1"]
    t_modern_case2 = modern_result["t_case2"]
    t_modern_other = max(
        0.0,
        t_modern - t_modern_sparsify - t_modern_case1 - t_modern_case2,
    )

    start = time.perf_counter()
    optimal_size = maximum_matching_size_networkx(left_nodes, right_nodes, edges)
    t_hopcroft = time.perf_counter() - start

    modern_estimate = modern_result["estimate"]
    modern_mu1 = modern_result["mu1"]
    modern_mu2 = modern_result["mu2"]
    selected_case = "case1" if modern_mu1 >= modern_mu2 else "case2"
    case1 = modern_result["case1"]
    case2 = modern_result["case2"]
    copied_num_vertices = (
        case1["copied_num_vertices"]
        if selected_case == "case1"
        else case2["copied_num_vertices"]
    )

    row: Dict[str, float | int | str] = {
        "graph_type": graph_type,
        **metrics,
        "expected_average_degree": reported_expected_degree(
            graph_type,
            n,
            expected_degree,
        ),
        "param": param,
        "trial": trial,
        "seed": seed,
        "optimal_size": optimal_size,
        "greedy_size": greedy_size,
        "random_greedy_size": random_greedy_size,
        "modern_estimate": modern_estimate,
        "modern_mu1": modern_mu1,
        "modern_mu2": modern_mu2,
        "selected_case": selected_case,
        "M_size": modern_result["M_size"],
        "Mprime_estimate": case1["Mprime_estimate"],
        "B1_estimate": case1["B1_estimate"],
        "B2_estimate": case2["B2_estimate"],
        "unmatched_num_vertices": case1["unmatched_num_vertices"],
        "copied_num_vertices": copied_num_vertices,
        "case1_copied_num_vertices": case1["copied_num_vertices"],
        "case2_copied_num_vertices": case2["copied_num_vertices"],
        "epsilon": modern_result["epsilon"],
        "k": modern_result["k"],
        "sparsify_c": modern_result["sparsify_c"],
        "greedy_ratio": ratio(greedy_size, optimal_size),
        "random_greedy_ratio": ratio(random_greedy_size, optimal_size),
        "modern_ratio": ratio(modern_estimate, optimal_size),
        "modern_error": modern_estimate - optimal_size,
        "modern_relative_error": relative_error(modern_estimate, optimal_size),
        "t_greedy": t_greedy,
        "t_random_greedy": t_random_greedy,
        "t_modern": t_modern,
        "t_modern_sparsify": t_modern_sparsify,
        "t_modern_case1": t_modern_case1,
        "t_modern_case2": t_modern_case2,
        "t_modern_other": t_modern_other,
        "t_hopcroft": t_hopcroft,
    }
    return row


def benchmark_rows(args: argparse.Namespace):
    n_values = parse_number_list(args.n_values, int)
    expected_degrees = parse_number_list(args.expected_degrees, float)
    graph_types = parse_number_list(args.graph_types, str)
    epsilons = parse_number_list(args.epsilons, float)

    for epsilon in epsilons:
        for graph_type in graph_types:
            degrees = [0.0] if graph_type == "sqrtreg" else expected_degrees
            for n in n_values:
                for expected_degree in degrees:
                    for trial in range(args.trials):
                        yield run_trial(
                            graph_type=graph_type,
                            n=n,
                            expected_degree=expected_degree,
                            trial=trial,
                            epsilon=epsilon,
                            k=args.k,
                            alpha_left=args.alpha_left,
                            alpha_right=args.alpha_right,
                            base_seed=args.base_seed,
                        )


def write_benchmark(args: argparse.Namespace) -> None:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if args.append else "w"
    write_header = not args.append or not output_path.exists() or output_path.stat().st_size == 0

    rows_written = 0
    with output_path.open(mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        for row in benchmark_rows(args):
            writer.writerow(row)
            rows_written += 1
            if args.progress:
                print(
                    f"wrote {rows_written:4d}: "
                    f"{row['graph_type']} n={row['n_left']} "
                    f"d={row['expected_average_degree']} eps={row['epsilon']} "
                    f"trial={row['trial']} modern_ratio={row['modern_ratio']:.4f}"
                )

    print(f"wrote {rows_written} rows to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run benchmark trials and log core matching metrics to CSV."
    )
    parser.add_argument(
        "--tier",
        choices=["tier1", "tier2", "custom"],
        default="tier1",
        help=(
            "Benchmark preset. tier1 is the main algorithm comparison; "
            "tier2 is the epsilon sensitivity sweep."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="CSV output path. Defaults depend on --tier.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to the output CSV instead of overwriting it.",
    )
    parser.add_argument(
        "--graph-types",
        default="er,powerlaw,sqrtreg",
        help="Comma-separated graph families. Default: er,powerlaw,sqrtreg",
    )
    parser.add_argument(
        "--n-values",
        default=None,
        help="Comma-separated balanced side sizes. Defaults depend on --tier.",
    )
    parser.add_argument(
        "--expected-degrees",
        default=None,
        help=(
            "Comma-separated expected average degrees for ER/power-law graphs. "
            "The generated ER probability is p=d/n. Defaults depend on --tier."
        ),
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Trials per graph family/parameter. Defaults depend on --tier.",
    )
    parser.add_argument(
        "--epsilons",
        default=None,
        help="Comma-separated epsilon values. Defaults depend on --tier.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Explicit copied-graph capacity parameter. Overrides epsilon-derived k.",
    )
    parser.add_argument(
        "--alpha-left",
        type=float,
        default=2.5,
        help="Power-law exponent for left vertices. Default: 2.5",
    )
    parser.add_argument(
        "--alpha-right",
        type=float,
        default=2.5,
        help="Power-law exponent for right vertices. Default: 2.5",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=787,
        help="Base seed used to derive per-trial seeds. Default: 787",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print one progress line per completed trial.",
    )
    return parser


def apply_tier_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.tier == "tier1":
        args.output = args.output or "results/tier1_main.csv"
        args.n_values = args.n_values or DEFAULT_TIER1_N_VALUES
        args.expected_degrees = args.expected_degrees or DEFAULT_TIER1_DEGREES
        args.trials = args.trials or DEFAULT_TIER1_TRIALS
        args.epsilons = args.epsilons or str(DEFAULT_TIER1_EPSILON)
    elif args.tier == "tier2":
        args.output = args.output or "results/tier2_epsilon_sweep.csv"
        args.n_values = args.n_values or DEFAULT_TIER2_N_VALUES
        args.expected_degrees = args.expected_degrees or DEFAULT_TIER2_DEGREES
        args.trials = args.trials or DEFAULT_TIER2_TRIALS
        args.epsilons = args.epsilons or DEFAULT_TIER2_EPSILONS
    else:
        args.output = args.output or DEFAULT_OUTPUT
        args.n_values = args.n_values or DEFAULT_TIER1_N_VALUES
        args.expected_degrees = args.expected_degrees or DEFAULT_TIER1_DEGREES
        args.trials = args.trials or DEFAULT_TIER1_TRIALS
        args.epsilons = args.epsilons or str(DEFAULT_PAPER_EPSILON)

    return args


def main() -> None:
    parser = build_parser()
    args = apply_tier_defaults(parser.parse_args())

    if args.trials <= 0:
        parser.error("--trials must be positive")
    for epsilon in parse_number_list(args.epsilons, float):
        if not (0.0 < epsilon < 1.0):
            parser.error("--epsilons must contain values in (0, 1)")
    if args.k is not None and args.k <= 0:
        parser.error("--k must be positive")

    write_benchmark(args)


if __name__ == "__main__":
    main()
