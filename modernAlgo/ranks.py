from __future__ import annotations

import hashlib
import random
from typing import Dict, Hashable, Protocol, Tuple

Node = Hashable
Edge = Tuple[Node, Node]


def canonical_edge(edge: Edge) -> Edge:
    """
    Return a canonical representation of an undirected edge.

    Example:
        (5, 2) -> (2, 5)

    This ensures that:
    - the same undirected edge always gets the same key
    - lazy rank assignment is stable
    """
    u, v = edge
    return (u, v) if u <= v else (v, u)


class EdgeRanks(Protocol):
    """
    Protocol for edge-rank providers.
    """

    def rank(self, edge: Edge) -> float:
        ...

    def order_key(self, edge: Edge) -> tuple[float, Edge]:
        ...

    def is_before(self, edge1: Edge, edge2: Edge) -> bool:
        ...


class LazyEdgeRanks:
    """
    Lazy random ranks for undirected edges.

    One instance of this class corresponds to one random permutation/order
    over the edges, represented by i.i.d. random ranks in (0,1).

    The ranks are:
    - assigned lazily on first access
    - memoized thereafter

    We also define a total order on edges using:
        (rank(edge), canonical_edge(edge))
    so ties are broken deterministically.

    Notes
    -----
    In theory, continuous random ranks tie with probability zero.
    We still add a deterministic tie-breaker for robustness.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._ranks: Dict[Edge, float] = {}

    def rank(self, edge: Edge) -> float:
        """
        Return the lazy random rank of an edge.
        """
        e = canonical_edge(edge)
        if e not in self._ranks:
            self._ranks[e] = self._rng.random()
        return self._ranks[e]

    def order_key(self, edge: Edge) -> tuple[float, Edge]:
        """
        Return a total-order key for the edge.

        Edges are ordered first by rank, then lexicographically by
        canonical edge representation.
        """
        e = canonical_edge(edge)
        return (self.rank(e), e)

    def is_before(self, edge1: Edge, edge2: Edge) -> bool:
        """
        Return True iff edge1 appears before edge2 in the induced order.
        """
        return self.order_key(edge1) < self.order_key(edge2)

    def clear(self) -> None:
        """
        Clear all cached ranks. Mostly useful for testing.
        """
        self._ranks.clear()


class HashEdgeRanks:
    """
    Query-order independent random ranks for undirected edges.

    The rank of an edge is derived from (seed, canonical_edge(edge)) using a
    stable cryptographic hash. This gives each edge a deterministic pseudo-
    random rank in [0, 1), independent of the order in which edges are queried.
    """

    def __init__(self, seed: int | str | None = None) -> None:
        self.seed = "" if seed is None else str(seed)
        self._ranks: Dict[Edge, float] = {}

    def rank(self, edge: Edge) -> float:
        """
        Return the deterministic pseudo-random rank of an edge.
        """
        e = canonical_edge(edge)
        if e not in self._ranks:
            payload = repr((self.seed, e)).encode("utf-8")
            digest = hashlib.blake2b(payload, digest_size=8).digest()
            value = int.from_bytes(digest, byteorder="big", signed=False)
            self._ranks[e] = value / 2**64

        return self._ranks[e]

    def order_key(self, edge: Edge) -> tuple[float, Edge]:
        """
        Return a total-order key for the edge.
        """
        e = canonical_edge(edge)
        return (self.rank(e), e)

    def is_before(self, edge1: Edge, edge2: Edge) -> bool:
        """
        Return True iff edge1 appears before edge2 in the induced order.
        """
        return self.order_key(edge1) < self.order_key(edge2)

    def clear(self) -> None:
        """
        Clear cached rank values.
        """
        self._ranks.clear()
