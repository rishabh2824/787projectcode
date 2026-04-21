from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Hashable, Iterable, Protocol, Tuple

from modernAlgo.behnezhad_rgmm_oracle import BehnezhadRGMMOracle
from modernAlgo.case1.view_copied import Case1CopiedView
from modernAlgo.case1.view_unmatched import UnmatchedInducedView
from modernAlgo.case2.copied_graph import Case2CopiedView
from modernAlgo.ranks import HashEdgeRanks, canonical_edge
from modernAlgo.rgmm_oracle import RGMMOracle
from greedy import greedy_bipartite_matching

Node = Hashable
Edge = Tuple[Node, Node]


class ValidatedGraphView(Protocol):
    def num_vertices(self) -> int:
        ...

    def vertex_at(self, index: int) -> Node:
        ...


@dataclass
class ValidationResult:
    name: str
    num_vertices: int
    mismatches: list[tuple[Node, bool, bool, Edge | None, Edge | None]]
    beh_edge_queries: int
    beh_lowest_queries: int
    beh_expose_next_calls: int
    beh_sampled_indices: int

    @property
    def passed(self) -> bool:
        return not self.mismatches


@dataclass
class InvariantResult:
    name: str
    num_vertices: int
    errors: list[str]
    beh_edge_queries: int
    beh_lowest_queries: int
    beh_expose_next_calls: int
    beh_sampled_indices: int

    @property
    def passed(self) -> bool:
        return not self.errors


def all_vertices(view: ValidatedGraphView) -> list[Node]:
    return [view.vertex_at(index) for index in range(view.num_vertices())]


def validate_view(
    name: str,
    view: ValidatedGraphView,
    seed: int = 123,
) -> ValidationResult:
    ranks = HashEdgeRanks(seed=seed)
    exact = RGMMOracle(view, ranks)
    behnezhad = BehnezhadRGMMOracle(
        view,
        seed=seed,
        validation_rank_provider=ranks,
    )

    mismatches: list[tuple[Node, bool, bool, Edge | None, Edge | None]] = []
    for vertex in all_vertices(view):
        exact_matched = exact.vertex_matched(vertex)
        beh_matched = behnezhad.vertex_matched(vertex)
        exact_edge = exact.matched_edge(vertex)
        beh_edge = behnezhad.matched_edge(vertex)

        if exact_edge is not None:
            exact_edge = canonical_edge(exact_edge)
        if beh_edge is not None:
            beh_edge = canonical_edge(beh_edge)

        if exact_matched != beh_matched or exact_edge != beh_edge:
            mismatches.append(
                (vertex, exact_matched, beh_matched, exact_edge, beh_edge)
            )

    return ValidationResult(
        name=name,
        num_vertices=view.num_vertices(),
        mismatches=mismatches,
        beh_edge_queries=behnezhad.edge_queries,
        beh_lowest_queries=behnezhad.lowest_queries,
        beh_expose_next_calls=behnezhad.expose_next_calls,
        beh_sampled_indices=behnezhad.sampled_indices,
    )


def validate_normal_mode_invariants(
    name: str,
    view: ValidatedGraphView,
    seed: int = 123,
) -> InvariantResult:
    oracle = BehnezhadRGMMOracle(view, seed=seed)
    vertices = all_vertices(view)
    errors: list[str] = []
    matched_by_vertex: dict[Node, Edge | None] = {}

    for vertex in vertices:
        matched = oracle.vertex_matched(vertex)
        edge = oracle.matched_edge(vertex)
        if edge is not None:
            edge = canonical_edge(edge)

        matched_by_vertex[vertex] = edge
        if matched != (edge is not None):
            errors.append(
                f"vertex_matched/matched_edge disagree for {vertex!r}: "
                f"{matched=} {edge=}"
            )
        if edge is not None and vertex not in edge:
            errors.append(f"matched_edge for {vertex!r} does not contain vertex: {edge!r}")

    unique_edges = {edge for edge in matched_by_vertex.values() if edge is not None}
    endpoint_to_edge: dict[Node, Edge] = {}
    for edge in unique_edges:
        u, v = edge
        for endpoint in (u, v):
            if endpoint in endpoint_to_edge and endpoint_to_edge[endpoint] != edge:
                errors.append(
                    f"endpoint {endpoint!r} appears in multiple matched edges: "
                    f"{endpoint_to_edge[endpoint]!r} and {edge!r}"
                )
            endpoint_to_edge[endpoint] = edge

    for edge in unique_edges:
        u, v = edge
        if u in matched_by_vertex and matched_by_vertex[u] != edge:
            errors.append(f"endpoint {u!r} does not report matched edge {edge!r}")
        if v in matched_by_vertex and matched_by_vertex[v] != edge:
            errors.append(f"endpoint {v!r} does not report matched edge {edge!r}")

    return InvariantResult(
        name=name,
        num_vertices=view.num_vertices(),
        errors=errors,
        beh_edge_queries=oracle.edge_queries,
        beh_lowest_queries=oracle.lowest_queries,
        beh_expose_next_calls=oracle.expose_next_calls,
        beh_sampled_indices=oracle.sampled_indices,
    )


def build_validation_views() -> Iterable[tuple[str, ValidatedGraphView]]:
    left_nodes = [0, 1, 2, 3]
    right_nodes = [10, 11, 12, 13]
    edges = [
        (0, 10),
        (0, 11),
        (1, 10),
        (1, 12),
        (2, 11),
        (2, 13),
        (3, 12),
    ]

    yield (
        "unmatched_induced",
        UnmatchedInducedView(
            left_nodes=left_nodes,
            right_nodes=right_nodes,
            edges=edges,
            M=[(0, 10), (1, 12)],
        ),
    )

    yield (
        "case2_copied",
        Case2CopiedView(
            left_nodes=left_nodes,
            right_nodes=right_nodes,
            edges=edges,
            M=[(0, 10), (1, 12), (2, 11)],
            k=2,
        ),
    )

    class FakeInnerOracle:
        def __init__(self, matched_vertices: Iterable[Node]) -> None:
            self._matched_vertices = set(matched_vertices)

        def vertex_matched(self, v: Node) -> bool:
            return v in self._matched_vertices

    yield (
        "case1_copied",
        Case1CopiedView(
            left_nodes=left_nodes,
            right_nodes=right_nodes,
            edges=edges,
            M=[(0, 10), (1, 12)],
            inner_mprime_oracle=FakeInnerOracle({2, 11}),
            k=2,
        ),
    )

    dense_left_nodes = [0, 1, 2]
    dense_right_nodes = [10, 11, 12]
    dense_edges = [
        (u, v)
        for u in dense_left_nodes
        for v in dense_right_nodes
    ]

    yield (
        "dense_unmatched_induced",
        UnmatchedInducedView(
            left_nodes=dense_left_nodes,
            right_nodes=dense_right_nodes,
            edges=dense_edges,
            M=[(0, 10)],
        ),
    )

    yield (
        "dense_case2_copied",
        Case2CopiedView(
            left_nodes=dense_left_nodes,
            right_nodes=dense_right_nodes,
            edges=dense_edges,
            M=[(0, 10), (1, 11)],
            k=2,
        ),
    )


def generate_bipartite_graph(
    n_left: int,
    n_right: int,
    p: float,
    seed: int,
) -> tuple[list[Node], list[Node], list[Edge]]:
    rng = random.Random(seed)
    left_nodes = list(range(n_left))
    right_nodes = list(range(n_left, n_left + n_right))
    edges = [
        (u, v)
        for u in left_nodes
        for v in right_nodes
        if rng.random() < p
    ]
    return left_nodes, right_nodes, edges


def build_random_validation_views() -> Iterable[tuple[str, ValidatedGraphView]]:
    graph_specs = [
        (3, 3, 0.35),
        (4, 4, 0.45),
        (5, 4, 0.55),
        (5, 5, 0.75),
    ]

    for graph_index, (n_left, n_right, p) in enumerate(graph_specs):
        for graph_seed in range(5):
            left_nodes, right_nodes, edges = generate_bipartite_graph(
                n_left=n_left,
                n_right=n_right,
                p=p,
                seed=10_000 * graph_index + graph_seed,
            )
            if not edges:
                continue

            matching = greedy_bipartite_matching(left_nodes, right_nodes, edges)
            partial_matching = matching[: max(0, len(matching) // 2)]

            unmatched_view = UnmatchedInducedView(
                left_nodes=left_nodes,
                right_nodes=right_nodes,
                edges=edges,
                M=partial_matching,
            )
            yield (
                f"random_unmatched[g={graph_index},seed={graph_seed}]",
                unmatched_view,
            )

            for k in (1, 2, 3):
                yield (
                    f"random_case2[g={graph_index},seed={graph_seed},k={k}]",
                    Case2CopiedView(
                        left_nodes=left_nodes,
                        right_nodes=right_nodes,
                        edges=edges,
                        M=partial_matching,
                        k=k,
                    ),
                )

            inner_exact = RGMMOracle(
                unmatched_view,
                HashEdgeRanks(seed=f"inner-{graph_index}-{graph_seed}"),
            )

            yield (
                f"random_case1[g={graph_index},seed={graph_seed}]",
                Case1CopiedView(
                    left_nodes=left_nodes,
                    right_nodes=right_nodes,
                    edges=edges,
                    M=partial_matching,
                    inner_mprime_oracle=inner_exact,
                    k=2,
                ),
            )


def main() -> None:
    seeds = [7, 123, 999]
    views = list(build_validation_views()) + list(build_random_validation_views())
    results = [
        validate_view(f"{name}[seed={seed}]", view, seed=seed)
        for seed in seeds
        for name, view in views
    ]
    invariant_results = [
        validate_normal_mode_invariants(f"{name}[seed={seed}]", view, seed=seed)
        for seed in seeds
        for name, view in views
    ]

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(
            f"{status} {result.name}: vertices={result.num_vertices}, "
            f"mismatches={len(result.mismatches)}, "
            f"edge_queries={result.beh_edge_queries}, "
            f"lowest={result.beh_lowest_queries}, "
            f"expose_next={result.beh_expose_next_calls}, "
            f"sampled_indices={result.beh_sampled_indices}"
        )

        for vertex, exact_matched, beh_matched, exact_edge, beh_edge in result.mismatches[:10]:
            print(
                "  mismatch "
                f"vertex={vertex!r}, exact=({exact_matched}, {exact_edge!r}), "
                f"behnezhad=({beh_matched}, {beh_edge!r})"
            )

    for result in invariant_results:
        status = "PASS" if result.passed else "FAIL"
        print(
            f"{status} invariant {result.name}: vertices={result.num_vertices}, "
            f"errors={len(result.errors)}, "
            f"edge_queries={result.beh_edge_queries}, "
            f"lowest={result.beh_lowest_queries}, "
            f"expose_next={result.beh_expose_next_calls}, "
            f"sampled_indices={result.beh_sampled_indices}"
        )

        for error in result.errors[:10]:
            print(f"  {error}")

    if not all(result.passed for result in results):
        raise SystemExit(1)
    if not all(result.passed for result in invariant_results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
