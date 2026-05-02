from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt

DEFAULT_INPUT = "results/raw_trials.csv"
DEFAULT_OUTPUT_DIR = "figures"

SERIES_STYLES = {
    "greedy": {
        "label": "Greedy",
        "color": "#1f77b4",
        "marker": "o",
    },
    "random_greedy": {
        "label": "Randomized Greedy",
        "color": "#ff7f0e",
        "marker": "s",
    },
    "modern": {
        "label": "Modern",
        "color": "#2ca02c",
        "marker": "^",
    },
    "hopcroft": {
        "label": "Hopcroft-Karp",
        "color": "#4d4d4d",
        "marker": "D",
    },
}

MODERN_B = 1.0 + 2.0**0.5
IGNORED_EPSILONS = {0.1}


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: Dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value != "" else 0.0


def epsilon_filtered_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    return [row for row in rows if as_float(row, "epsilon") not in IGNORED_EPSILONS]


def graph_families(rows: Sequence[Dict[str, str]]) -> List[str]:
    return sorted({row["graph_type"] for row in rows})


def mean_by_n(rows: Iterable[Dict[str, str]], key: str) -> Dict[int, float]:
    groups: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        groups[int(row["n_left"])].append(as_float(row, key))

    return {n: mean(values) for n, values in groups.items()}


def mean_by_float_key(rows: Iterable[Dict[str, str]], group_key: str, value_key: str) -> Dict[float, float]:
    groups: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        groups[as_float(row, group_key)].append(as_float(row, value_key))

    return {x_value: mean(values) for x_value, values in groups.items()}


def density_axis_key(rows: Sequence[Dict[str, str]]) -> str:
    if rows and "expected_average_degree" in rows[0]:
        return "expected_average_degree"
    return "param"


def density_axis_label(rows: Sequence[Dict[str, str]]) -> str:
    if density_axis_key(rows) == "expected_average_degree":
        return "expected average degree"
    return "edge probability / density parameter"


def grouped_means(
    rows: Iterable[Dict[str, str]],
    group_key: str,
    value_keys: Sequence[str],
) -> Dict[float, Dict[str, float]]:
    grouped_rows: dict[float, list[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped_rows[as_float(row, group_key)].append(row)

    output: Dict[float, Dict[str, float]] = {}
    for x_value, group in grouped_rows.items():
        output[x_value] = {
            value_key: mean(as_float(row, value_key) for row in group)
            for value_key in value_keys
        }
    return output


def case1_fraction_by_grid(rows: Sequence[Dict[str, str]]) -> Dict[tuple[int, float], float]:
    y_key = density_axis_key(rows)
    groups: dict[tuple[int, float], list[str]] = defaultdict(list)
    for row in rows:
        key = (int(row["n_left"]), as_float(row, y_key))
        groups[key].append(row["selected_case"])

    return {
        key: sum(1 for selected in selected_cases if selected == "case1") / len(selected_cases)
        for key, selected_cases in groups.items()
    }


def graph_label(graph_type: str) -> str:
    labels = {
        "er": "Erdos-Renyi",
        "powerlaw": "Power-law",
        "sqrtreg": "Near-regular sqrt(n)",
    }
    return labels.get(graph_type, graph_type)


def configure_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.45, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(frameon=False)


def save_figure(fig, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_approximation_by_graph(
    rows: Sequence[Dict[str, str]],
    graph_type: str,
    output_dir: Path,
) -> None:
    graph_rows = [row for row in rows if row["graph_type"] == graph_type]
    if not graph_rows:
        return

    series = {
        "greedy": mean_by_n(graph_rows, "greedy_ratio"),
        "random_greedy": mean_by_n(graph_rows, "random_greedy_ratio"),
        "modern": mean_by_n(graph_rows, "modern_ratio"),
    }
    n_values = sorted(series["greedy"])

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for name, values_by_n in series.items():
        style = SERIES_STYLES[name]
        y_values = [values_by_n[n] for n in n_values]
        ax.plot(
            n_values,
            y_values,
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            linewidth=2.0,
            markersize=5.0,
        )

    hopcroft_style = SERIES_STYLES["hopcroft"]
    ax.plot(
        n_values,
        [1.0 for _ in n_values],
        label="Hopcroft-Karp (OPT)",
        color=hopcroft_style["color"],
        linestyle="--",
        linewidth=1.8,
    )

    ax.set_ylim(0.75, 1.05)
    configure_axes(
        ax,
        title=f"Approximation Ratio vs Graph Size ({graph_label(graph_type)})",
        xlabel="n per side",
        ylabel="Approximation ratio",
    )
    save_figure(
        fig,
        output_dir,
        f"approx_ratio_vs_n_{graph_type}",
    )


def plot_runtime_by_graph(
    rows: Sequence[Dict[str, str]],
    graph_type: str,
    output_dir: Path,
) -> None:
    graph_rows = [row for row in rows if row["graph_type"] == graph_type]
    if not graph_rows:
        return

    series = {
        "greedy": mean_by_n(graph_rows, "t_greedy"),
        "random_greedy": mean_by_n(graph_rows, "t_random_greedy"),
        "modern": mean_by_n(graph_rows, "t_modern"),
        "hopcroft": mean_by_n(graph_rows, "t_hopcroft"),
    }
    n_values = sorted(series["greedy"])

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for name, values_by_n in series.items():
        style = SERIES_STYLES[name]
        y_values = [max(values_by_n[n], 1e-9) for n in n_values]
        ax.plot(
            n_values,
            y_values,
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            linewidth=2.0,
            markersize=5.0,
        )

    ax.set_yscale("log")
    configure_axes(
        ax,
        title=f"Runtime vs Graph Size ({graph_label(graph_type)})",
        xlabel="n per side",
        ylabel="Runtime (seconds, log scale)",
    )
    save_figure(
        fig,
        output_dir,
        f"runtime_vs_n_{graph_type}",
    )


def plot_approximation_by_density(
    rows: Sequence[Dict[str, str]],
    graph_type: str,
    output_dir: Path,
) -> None:
    graph_rows = [row for row in rows if row["graph_type"] == graph_type]
    if not graph_rows:
        return

    x_key = density_axis_key(graph_rows)
    series = {
        "greedy": mean_by_float_key(graph_rows, x_key, "greedy_ratio"),
        "random_greedy": mean_by_float_key(
            graph_rows,
            x_key,
            "random_greedy_ratio",
        ),
        "modern": mean_by_float_key(graph_rows, x_key, "modern_ratio"),
    }
    x_values = sorted(series["greedy"])

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for name, values_by_x in series.items():
        style = SERIES_STYLES[name]
        y_values = [values_by_x[x_value] for x_value in x_values]
        ax.plot(
            x_values,
            y_values,
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            linewidth=2.0,
            markersize=5.0,
        )

    ax.plot(
        x_values,
        [1.0 for _ in x_values],
        label="Hopcroft-Karp (OPT)",
        color=SERIES_STYLES["hopcroft"]["color"],
        linestyle="--",
        linewidth=1.8,
    )

    ax.set_ylim(0.75, 1.05)
    configure_axes(
        ax,
        title=f"Approximation Ratio vs Density ({graph_label(graph_type)})",
        xlabel=density_axis_label(graph_rows),
        ylabel="Approximation ratio",
    )
    save_figure(
        fig,
        output_dir,
        f"approx_ratio_vs_density_{graph_type}",
    )


def plot_modern_decomposition(
    rows: Sequence[Dict[str, str]],
    graph_type: str,
    output_dir: Path,
) -> None:
    graph_rows = [row for row in rows if row["graph_type"] == graph_type]
    if not graph_rows:
        return

    grouped = grouped_means(
        graph_rows,
        "n_left",
        [
            "M_size",
            "Mprime_estimate",
            "B1_estimate",
            "B2_estimate",
            "k",
        ],
    )
    n_values = sorted(grouped)

    m_values = [grouped[n]["M_size"] for n in n_values]
    mprime_values = [
        (1.0 - 1.0 / MODERN_B) * grouped[n]["Mprime_estimate"]
        for n in n_values
    ]
    b1_values = [
        grouped[n]["B1_estimate"] / (grouped[n]["k"] * MODERN_B)
        if grouped[n]["k"] > 0
        else 0.0
        for n in n_values
    ]
    b2_values = [
        grouped[n]["B2_estimate"] / (grouped[n]["k"] * MODERN_B)
        if grouped[n]["k"] > 0
        else 0.0
        for n in n_values
    ]

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    bar_width = 0.62
    ax.bar(
        n_values,
        m_values,
        width=bar_width,
        label="|M|",
        color="#4c78a8",
    )
    ax.bar(
        n_values,
        mprime_values,
        width=bar_width,
        bottom=m_values,
        label="(1 - 1/b) M'_est",
        color="#f58518",
    )
    bottom_b1 = [m + mp for m, mp in zip(m_values, mprime_values)]
    ax.bar(
        n_values,
        b1_values,
        width=bar_width,
        bottom=bottom_b1,
        label="B1_est / (kb)",
        color="#54a24b",
    )
    bottom_b2 = [m + mp + b1 for m, mp, b1 in zip(m_values, mprime_values, b1_values)]
    ax.bar(
        n_values,
        b2_values,
        width=bar_width,
        bottom=bottom_b2,
        label="B2_est / (kb)",
        color="#e45756",
    )

    configure_axes(
        ax,
        title=f"Modern Estimate Decomposition ({graph_label(graph_type)})",
        xlabel="n per side",
        ylabel="Mean contribution",
    )
    save_figure(
        fig,
        output_dir,
        f"modern_decomposition_{graph_type}",
    )


def plot_case_selection_heatmap(
    rows: Sequence[Dict[str, str]],
    graph_type: str,
    output_dir: Path,
) -> None:
    graph_rows = [row for row in rows if row["graph_type"] == graph_type]
    if not graph_rows:
        return

    n_values = sorted({int(row["n_left"]) for row in graph_rows})
    y_key = density_axis_key(graph_rows)
    density_values = sorted({as_float(row, y_key) for row in graph_rows})
    fractions = case1_fraction_by_grid(graph_rows)
    matrix = [
        [fractions.get((n_value, density), 0.0) for n_value in n_values]
        for density in density_values
    ]

    fig_width = max(6.2, 0.7 * len(n_values) + 2.2)
    fig_height = max(4.2, 0.55 * len(density_values) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")

    ax.set_xticks(range(len(n_values)), labels=[str(n) for n in n_values])
    ax.set_yticks(
        range(len(density_values)),
        labels=[f"{density:g}" for density in density_values],
    )
    ax.set_xlabel("n per side")
    ax.set_ylabel(density_axis_label(graph_rows))
    ax.set_title(f"Case 1 Selection Rate ({graph_label(graph_type)})")

    for y_index, density in enumerate(density_values):
        for x_index, n_value in enumerate(n_values):
            value = fractions.get((n_value, density), 0.0)
            text_color = "white" if value < 0.45 else "black"
            ax.text(
                x_index,
                y_index,
                f"{100.0 * value:.0f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("fraction of trials selecting Case 1")
    save_figure(
        fig,
        output_dir,
        f"case_selection_heatmap_{graph_type}",
    )


def has_runtime_breakdown(rows: Sequence[Dict[str, str]]) -> bool:
    required = {
        "t_modern_sparsify",
        "t_modern_case1",
        "t_modern_case2",
        "t_modern_other",
    }
    return bool(rows) and required.issubset(rows[0].keys())


def plot_runtime_breakdown(
    rows: Sequence[Dict[str, str]],
    graph_type: str,
    output_dir: Path,
) -> None:
    graph_rows = [row for row in rows if row["graph_type"] == graph_type]
    if not graph_rows or not has_runtime_breakdown(graph_rows):
        return

    grouped = grouped_means(
        graph_rows,
        "n_left",
        [
            "t_modern_sparsify",
            "t_modern_case1",
            "t_modern_case2",
            "t_modern_other",
        ],
    )
    n_values = sorted(grouped)
    sparsify_values = [grouped[n]["t_modern_sparsify"] for n in n_values]
    case1_values = [grouped[n]["t_modern_case1"] for n in n_values]
    case2_values = [grouped[n]["t_modern_case2"] for n in n_values]
    other_values = [grouped[n]["t_modern_other"] for n in n_values]

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    bar_width = 0.62
    ax.bar(
        n_values,
        sparsify_values,
        width=bar_width,
        label="Sparsification",
        color="#4c78a8",
    )
    ax.bar(
        n_values,
        case1_values,
        width=bar_width,
        bottom=sparsify_values,
        label="Case 1",
        color="#f58518",
    )
    bottom_case2 = [s + c1 for s, c1 in zip(sparsify_values, case1_values)]
    ax.bar(
        n_values,
        case2_values,
        width=bar_width,
        bottom=bottom_case2,
        label="Case 2",
        color="#54a24b",
    )
    bottom_other = [
        s + c1 + c2
        for s, c1, c2 in zip(sparsify_values, case1_values, case2_values)
    ]
    ax.bar(
        n_values,
        other_values,
        width=bar_width,
        bottom=bottom_other,
        label="Other overhead",
        color="#b279a2",
    )

    configure_axes(
        ax,
        title=f"Modern Runtime Breakdown ({graph_label(graph_type)})",
        xlabel="n per side",
        ylabel="Mean runtime (seconds)",
    )
    save_figure(
        fig,
        output_dir,
        f"modern_runtime_breakdown_{graph_type}",
    )


def plot_error_distribution(
    rows: Sequence[Dict[str, str]],
    output_dir: Path,
) -> None:
    families = graph_families(rows)
    if not families:
        return

    metric_keys = ["greedy_ratio", "random_greedy_ratio", "modern_ratio"]
    labels = ["Greedy", "Randomized\nGreedy", "Modern"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig_width = max(7.2, 1.25 * len(families) * len(metric_keys))
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))

    positions = []
    box_data = []
    box_colors = []
    center_positions = []
    center_labels = []

    base_position = 1.0
    within_gap = 0.75
    family_gap = 1.25
    for family in families:
        family_rows = [row for row in rows if row["graph_type"] == family]
        family_positions = []
        for metric_index, metric_key in enumerate(metric_keys):
            position = base_position + metric_index * within_gap
            values = [as_float(row, metric_key) for row in family_rows]
            positions.append(position)
            box_data.append(values)
            box_colors.append(colors[metric_index])
            family_positions.append(position)

        center_positions.append(mean(family_positions))
        center_labels.append(graph_label(family))
        base_position += len(metric_keys) * within_gap + family_gap

    boxplot = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.48,
        patch_artist=True,
        showfliers=True,
    )
    for patch, color in zip(boxplot["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    for median in boxplot["medians"]:
        median.set_color("#222222")
        median.set_linewidth(1.4)

    legend_handles = [
        plt.Line2D([0], [0], color=color, lw=7, alpha=0.72, label=label.replace("\n", " "))
        for color, label in zip(colors, labels)
    ]
    ax.legend(handles=legend_handles, frameon=False)
    ax.set_xticks(center_positions, center_labels)
    ax.set_ylim(0.6, 1.0)
    configure_axes(
        ax,
        title="Approximation Ratio Distribution by Graph Family",
        xlabel="graph family",
        ylabel="approximation ratio",
    )
    save_figure(fig, output_dir, "ratio_distribution_by_family")


def plot_estimate_error_vs_opt(
    rows: Sequence[Dict[str, str]],
    output_dir: Path,
) -> None:
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for graph_type in graph_families(rows):
        graph_rows = [row for row in rows if row["graph_type"] == graph_type]
        ax.scatter(
            [as_float(row, "optimal_size") for row in graph_rows],
            [as_float(row, "modern_error") for row in graph_rows],
            label=graph_label(graph_type),
            alpha=0.72,
            s=34,
        )

    ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1.4)
    configure_axes(
        ax,
        title="Modern Estimate Error vs OPT",
        xlabel="OPT",
        ylabel="modern_estimate - OPT",
    )
    save_figure(fig, output_dir, "modern_error_vs_opt")


def plot_epsilon_sensitivity(
    rows: Sequence[Dict[str, str]],
    output_dir: Path,
) -> None:
    rows = epsilon_filtered_rows(rows)
    epsilon_values = sorted({as_float(row, "epsilon") for row in rows})
    if len(epsilon_values) <= 1:
        return

    grouped = grouped_means(
        rows,
        "epsilon",
        [
            "modern_ratio",
            "t_modern",
            "k",
        ],
    )
    x_values = sorted(grouped)
    modern_ratios = [grouped[epsilon]["modern_ratio"] for epsilon in x_values]
    modern_runtimes = [grouped[epsilon]["t_modern"] for epsilon in x_values]
    k_values = [grouped[epsilon]["k"] for epsilon in x_values]

    fig, ax_left = plt.subplots(figsize=(7.2, 4.8))
    ratio_line = ax_left.plot(
        x_values,
        modern_ratios,
        color="#2ca02c",
        marker="^",
        linewidth=2.0,
        label="Modern ratio",
    )
    ax_left.set_xlabel("epsilon")
    ax_left.set_ylabel("approximation ratio")
    ax_left.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.45)

    ax_right = ax_left.twinx()
    runtime_line = ax_right.plot(
        x_values,
        modern_runtimes,
        color="#9467bd",
        marker="o",
        linewidth=2.0,
        label="Modern runtime",
    )
    k_line = ax_right.plot(
        x_values,
        k_values,
        color="#d62728",
        marker="s",
        linestyle="--",
        linewidth=1.8,
        label="Derived k",
    )
    ax_right.set_ylabel("runtime seconds / k")

    lines = ratio_line + runtime_line + k_line
    ax_left.legend(lines, [line.get_label() for line in lines], frameon=False)
    ax_left.set_title("Epsilon Sensitivity")
    save_figure(fig, output_dir, "epsilon_sensitivity")


def plot_epsilon_sensitivity_by_graph(
    rows: Sequence[Dict[str, str]],
    output_dir: Path,
) -> None:
    for graph_type in graph_families(rows):
        graph_rows = epsilon_filtered_rows(
            [row for row in rows if row["graph_type"] == graph_type]
        )
        epsilon_values = sorted({as_float(row, "epsilon") for row in graph_rows})
        if len(epsilon_values) <= 1:
            continue

        grouped = grouped_means(
            graph_rows,
            "epsilon",
            [
                "modern_ratio",
                "t_modern",
                "k",
            ],
        )
        x_values = sorted(grouped)
        modern_ratios = [grouped[epsilon]["modern_ratio"] for epsilon in x_values]
        modern_runtimes = [grouped[epsilon]["t_modern"] for epsilon in x_values]
        k_values = [grouped[epsilon]["k"] for epsilon in x_values]

        fig, ax_left = plt.subplots(figsize=(7.2, 4.8))
        ratio_line = ax_left.plot(
            x_values,
            modern_ratios,
            color="#2ca02c",
            marker="^",
            linewidth=2.0,
            label="Modern ratio",
        )
        ax_left.set_xlabel("epsilon")
        ax_left.set_ylabel("approximation ratio")
        ax_left.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.45)

        ax_right = ax_left.twinx()
        runtime_line = ax_right.plot(
            x_values,
            modern_runtimes,
            color="#9467bd",
            marker="o",
            linewidth=2.0,
            label="Modern runtime",
        )
        k_line = ax_right.plot(
            x_values,
            k_values,
            color="#d62728",
            marker="s",
            linestyle="--",
            linewidth=1.8,
            label="Derived k",
        )
        ax_right.set_ylabel("runtime seconds / k")

        lines = ratio_line + runtime_line + k_line
        ax_left.legend(lines, [line.get_label() for line in lines], frameon=False)
        ax_left.set_title(f"Epsilon Sensitivity ({graph_label(graph_type)})")
        save_figure(
            fig,
            output_dir,
            f"epsilon_sensitivity_{graph_type}",
        )


def generate_full_plots(rows: Sequence[Dict[str, str]], output_dir: Path) -> None:
    for graph_type in graph_families(rows):
        plot_approximation_by_graph(rows, graph_type, output_dir)
        plot_runtime_by_graph(rows, graph_type, output_dir)
        plot_approximation_by_density(rows, graph_type, output_dir)
        plot_modern_decomposition(rows, graph_type, output_dir)
        plot_case_selection_heatmap(rows, graph_type, output_dir)
        plot_runtime_breakdown(rows, graph_type, output_dir)

    plot_error_distribution(rows, output_dir)
    plot_estimate_error_vs_opt(rows, output_dir)
    plot_epsilon_sensitivity(rows, output_dir)
    plot_epsilon_sensitivity_by_graph(rows, output_dir)


def generate_epsilon_only_plots(rows: Sequence[Dict[str, str]], output_dir: Path) -> None:
    plot_epsilon_sensitivity(rows, output_dir)
    plot_epsilon_sensitivity_by_graph(rows, output_dir)


def generate_plots(input_path: Path, output_dir: Path, mode: str) -> None:
    rows = read_rows(input_path)
    if not rows:
        raise ValueError(f"no benchmark rows found in {input_path}")

    if mode == "full":
        generate_full_plots(rows, output_dir)
    elif mode == "epsilon-only":
        generate_epsilon_only_plots(rows, output_dir)
    else:
        raise ValueError(f"unknown plot mode: {mode}")

    print(f"wrote plots to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate benchmark plots from raw trial CSV output."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Raw benchmark CSV path. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated figures. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "epsilon-only"],
        default="full",
        help="Plot set to generate. Use 'epsilon-only' for Tier 2 epsilon sensitivity figures only.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    generate_plots(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
