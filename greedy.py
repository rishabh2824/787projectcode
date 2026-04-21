from __future__ import annotations

from typing import Hashable, Iterable, List, Sequence, Tuple, Set, Dict
import random

Node = Hashable
Edge = Tuple[Node, Node]


class GreedyBipartiteMatching:
    """
    Greedy maximal matching for a bipartite graph G = (U, V, E).

    The algorithm scans edges in order and adds an edge (u, v) if both
    endpoints are currently unmatched.
    """

    def __init__(
        self,
        left_nodes: Sequence[Node],
        right_nodes: Sequence[Node],
        edges: Iterable[Edge],
    ) -> None:
        self.U: List[Node] = list(left_nodes)
        self.V: List[Node] = list(right_nodes)
        self.edges: List[Edge] = list(edges)

    def maximum_greedy_matching(self) -> List[Edge]:
        """
        Return a greedy maximal matching as a list of edges.
        """
        matched_u: Set[Node] = set()
        matched_v: Set[Node] = set()
        matching: List[Edge] = []

        for u, v in self.edges:
            if u not in matched_u and v not in matched_v:
                matching.append((u, v))
                matched_u.add(u)
                matched_v.add(v)

        return matching

    def matching_size(self) -> int:
        """
        Return the size of the greedy matching.
        """
        return len(self.maximum_greedy_matching())

    def matching_dict(self) -> Dict[Node, Node]:
        """
        Return the matching in dictionary form, with both directions:
        u -> v and v -> u
        """
        result: Dict[Node, Node] = {}
        for u, v in self.maximum_greedy_matching():
            result[u] = v
            result[v] = u
        return result


def greedy_bipartite_matching(
    left_nodes: Sequence[Node],
    right_nodes: Sequence[Node],
    edges: Iterable[Edge],
) -> List[Edge]:
    """
    Functional wrapper returning a greedy maximal matching.
    """
    solver = GreedyBipartiteMatching(left_nodes, right_nodes, edges)
    return solver.maximum_greedy_matching()


def greedy_bipartite_matching_size(
    left_nodes: Sequence[Node],
    right_nodes: Sequence[Node],
    edges: Iterable[Edge],
) -> int:
    """
    Functional wrapper returning only the greedy matching size.
    """
    solver = GreedyBipartiteMatching(left_nodes, right_nodes, edges)
    return solver.matching_size()


def shuffled_greedy_bipartite_matching(
    left_nodes: Sequence[Node],
    right_nodes: Sequence[Node],
    edges: Iterable[Edge],
    seed: int | None = None,
) -> List[Edge]:
    """
    Greedy maximal matching after randomly shuffling edge order.
    """
    edge_list = list(edges)
    rng = random.Random(seed)
    rng.shuffle(edge_list)
    return greedy_bipartite_matching(left_nodes, right_nodes, edge_list)


def is_valid_matching(matching: Iterable[Edge]) -> bool:
    """
    Check that no two edges in the matching share an endpoint.
    """
    seen: Set[Node] = set()
    for u, v in matching:
        if u in seen or v in seen:
            return False
        seen.add(u)
        seen.add(v)
    return True


if __name__ == "__main__":
    U = [0, 1]
    V = [10, 11]
    E = [
        (0, 10),
        (0, 11),
        (1, 10),
    ]

    matching = greedy_bipartite_matching(U, V, E)
    print("Greedy matching:", matching)
    print("Size:", len(matching))
    print("Valid?", is_valid_matching(matching))