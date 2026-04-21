from __future__ import annotations
import networkx as nx
from networkx.algorithms.bipartite.matching import hopcroft_karp_matching


def maximum_matching_size_networkx(
    left_nodes: list[int],
    right_nodes: list[int],
    edges: list[tuple[int, int]],
) -> int:
    """
    Compute exact maximum bipartite matching size using the Hopcroft--Karp algorithm.

    Parameters
    ----------
    left_nodes : list[int]
        Vertices on the left side U.
    right_nodes : list[int]
        Vertices on the right side V.
    edges : list[tuple[int, int]]
        Edges (u, v) with u in left_nodes and v in right_nodes.

    Returns
    -------
    int
        Size of the maximum matching.
    """
    G = nx.Graph()
    G.add_nodes_from(left_nodes, bipartite=0)
    G.add_nodes_from(right_nodes, bipartite=1)
    G.add_edges_from(edges)

    matching = hopcroft_karp_matching(G, top_nodes=left_nodes)

    # NetworkX returns both directions in the dict, so divide by 2.
    return len(matching) // 2

