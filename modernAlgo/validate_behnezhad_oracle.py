from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable, Protocol, Tuple

from modernAlgo.behnezhad_rgmm_oracle import BehnezhadRGMMOracle
from modernAlgo.case1.view_copied import Case1CopiedView
from modernAlgo.case1.view_unmatched import UnmatchedInducedView
from modernAlgo.case2.copied_graph import Case2CopiedView
from modernAlgo.ranks import HashEdgeRanks, canonical_edge
from modernAlgo.rgmm_oracle import RGMMOracle

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


def main() -> None:
    seeds = [7, 123, 999]
    results = [
        validate_view(f"{name}[seed={seed}]", view, seed=seed)
        for seed in seeds
        for name, view in build_validation_views()
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

    if not all(result.passed for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
