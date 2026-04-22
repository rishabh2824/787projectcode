from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence

DEFAULT_INPUT = "results/raw_trials.csv"
DEFAULT_OUTPUT_DIR = "results"


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: Dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value != "" else 0.0


def group_by(rows: Iterable[Dict[str, str]], keys: Sequence[str]):
    groups: dict[tuple[str, ...], list[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return groups


def percent_case(rows: Sequence[Dict[str, str]], case_name: str) -> float:
    if not rows:
        return 0.0
    count = sum(1 for row in rows if row["selected_case"] == case_name)
    return 100.0 * count / len(rows)


def write_rows(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def graph_family_summary(rows: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    output = []
    for (graph_type,), group in sorted(group_by(rows, ["graph_type"]).items()):
        output.append(
            {
                "graph_family": graph_type,
                "num_trials": len(group),
                "mean_OPT": mean(as_float(row, "optimal_size") for row in group),
                "mean_greedy_ratio": mean(as_float(row, "greedy_ratio") for row in group),
                "mean_random_greedy_ratio": mean(
                    as_float(row, "random_greedy_ratio") for row in group
                ),
                "mean_modern_ratio": mean(as_float(row, "modern_ratio") for row in group),
                "mean_greedy_runtime": mean(as_float(row, "t_greedy") for row in group),
                "mean_modern_runtime": mean(as_float(row, "t_modern") for row in group),
                "mean_hopcroft_runtime": mean(as_float(row, "t_hopcroft") for row in group),
                "modern_selected_case1_pct": percent_case(group, "case1"),
                "modern_selected_case2_pct": percent_case(group, "case2"),
            }
        )
    return output


def scaling_summary(rows: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    output = []
    for (graph_type, n_left), group in sorted(
        group_by(rows, ["graph_type", "n_left"]).items(),
        key=lambda item: (item[0][0], int(item[0][1])),
    ):
        output.append(
            {
                "graph_family": graph_type,
                "n": int(n_left),
                "num_trials": len(group),
                "average_edges": mean(as_float(row, "num_edges") for row in group),
                "mean_greedy_ratio": mean(as_float(row, "greedy_ratio") for row in group),
                "mean_modern_ratio": mean(as_float(row, "modern_ratio") for row in group),
                "mean_greedy_runtime": mean(as_float(row, "t_greedy") for row in group),
                "mean_modern_runtime": mean(as_float(row, "t_modern") for row in group),
                "mean_hopcroft_runtime": mean(as_float(row, "t_hopcroft") for row in group),
            }
        )
    return output


def epsilon_summary(rows: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    output = []
    for (graph_type, epsilon, k), group in sorted(
        group_by(rows, ["graph_type", "epsilon", "k"]).items(),
        key=lambda item: (item[0][0], float(item[0][1]), int(float(item[0][2]))),
    ):
        output.append(
            {
                "graph_family": graph_type,
                "epsilon": float(epsilon),
                "derived_or_effective_k": int(float(k)),
                "num_trials": len(group),
                "mean_modern_ratio": mean(as_float(row, "modern_ratio") for row in group),
                "mean_modern_runtime": mean(as_float(row, "t_modern") for row in group),
                "mean_copied_vertex_count": mean(
                    as_float(row, "copied_num_vertices") for row in group
                ),
                "modern_selected_case1_pct": percent_case(group, "case1"),
                "modern_selected_case2_pct": percent_case(group, "case2"),
            }
        )
    return output


def write_summaries(input_path: Path, output_dir: Path) -> None:
    rows = read_rows(input_path)
    if not rows:
        raise ValueError(f"no benchmark rows found in {input_path}")

    family_rows = graph_family_summary(rows)
    scaling_rows = scaling_summary(rows)
    epsilon_rows = epsilon_summary(rows)

    write_rows(
        output_dir / "summary_by_family.csv",
        [
            "graph_family",
            "num_trials",
            "mean_OPT",
            "mean_greedy_ratio",
            "mean_random_greedy_ratio",
            "mean_modern_ratio",
            "mean_greedy_runtime",
            "mean_modern_runtime",
            "mean_hopcroft_runtime",
            "modern_selected_case1_pct",
            "modern_selected_case2_pct",
        ],
        family_rows,
    )

    write_rows(
        output_dir / "summary_by_n.csv",
        [
            "graph_family",
            "n",
            "num_trials",
            "average_edges",
            "mean_greedy_ratio",
            "mean_modern_ratio",
            "mean_greedy_runtime",
            "mean_modern_runtime",
            "mean_hopcroft_runtime",
        ],
        scaling_rows,
    )

    write_rows(
        output_dir / "summary_by_epsilon.csv",
        [
            "graph_family",
            "epsilon",
            "derived_or_effective_k",
            "num_trials",
            "mean_modern_ratio",
            "mean_modern_runtime",
            "mean_copied_vertex_count",
            "modern_selected_case1_pct",
            "modern_selected_case2_pct",
        ],
        epsilon_rows,
    )

    print(f"wrote {output_dir / 'summary_by_family.csv'}")
    print(f"wrote {output_dir / 'summary_by_n.csv'}")
    print(f"wrote {output_dir / 'summary_by_epsilon.csv'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build benchmark summary tables from raw trial CSV output."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Raw benchmark CSV path. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for summary CSVs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    write_summaries(Path(args.input), Path(args.output_dir))


if __name__ == "__main__":
    main()
