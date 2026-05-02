# CS787 Bipartite Matching Approximation Experiments

This repository implements and evaluates algorithms for estimating the maximum matching size in bipartite graphs. It was built for a CS787 project comparing simple greedy baselines, exact Hopcroft-Karp matching, and a modern oracle-based approximation algorithm inspired by the Behnezhad matching-estimation framework.

The central object is a bipartite graph \(G = (U, V, E)\). Experiments generate several families of bipartite graphs, compute an exact optimum with Hopcroft-Karp, and compare that optimum against:

- deterministic greedy maximal matching,
- randomized greedy maximal matching,
- the modern oracle estimator implemented in `modernAlgo/`.

The main research focus is the modern algorithm. It estimates the maximum matching size without explicitly constructing the very large copied graphs used in the theoretical construction.

## Repository Layout

```text
.
|-- benchmark.py                  # Main benchmark runner; writes raw trial CSVs
|-- experiment.py                 # Smaller console experiment driver
|-- greedy.py                     # Greedy maximal matching baseline
|-- hopcroft.py                   # Exact Hopcroft-Karp baseline using NetworkX
|-- plot_benchmark.py             # Plot generation from benchmark CSVs
|-- summarize_benchmark.py        # Summary-table generation from benchmark CSVs
|-- graphGenerators/
|   |-- ERgraph.py                # Erdos-Renyi bipartite graphs
|   |-- powerLaw.py               # Power-law-like bipartite graphs
|   `-- sqrt_regular.py           # Near-regular degree ~ sqrt(n) graphs
|-- modernAlgo/
|   |-- modern_algo.py            # Top-level modern algorithm pipeline
|   |-- sparsify.py               # Algorithm 2-style sparsification matching
|   |-- graph_oracle.py           # Adjacency-list oracle interface
|   |-- behnezhad_rgmm_oracle.py  # Local random greedy maximal matching oracle
|   |-- case1/                    # Case 1 estimator and lazy graph views
|   `-- case2/                    # Case 2 estimator and lazy graph views
|-- results/                      # Generated CSV results and summaries
`-- figures/                      # Generated plots
```

## Modern Algorithm

The modern implementation is exposed through:

```python
from modernAlgo.modern_algo import run_modern_oracle
```

Given `left_nodes`, `right_nodes`, and an edge list, `run_modern_oracle(...)` returns a dictionary containing the final estimate, the two case estimates, the sparsified matching size, the effective capacity parameter `k`, timing breakdowns, and internal estimator metadata.

At a high level, the algorithm follows this pipeline:

1. **Graph-oracle wrapper**

   `EdgeListBipartiteGraph` adapts the repository's explicit edge-list graphs into an adjacency-list oracle with:

   - `degree(v)`,
   - `neighbor_at(v, i)`,
   - partition accessors for left and right vertices.

   This lets the algorithm query graph neighborhoods locally instead of requiring all downstream structures to be materialized.

2. **Sparsification matching**

   `sparsify_partial_matching_from_graph` builds an initial partial matching `M`. For each currently unmatched vertex, it samples

   ```text
   c = ceil(2 * sqrt(n) * log n)
   ```

   random neighbors and adds the first edge whose other endpoint is also unmatched. This is the Algorithm 2-style sparsification step used by both later cases.

3. **Capacity parameter**

   The copied-graph capacity is controlled by `k`. If the user does not pass `--k`, the implementation derives it from epsilon:

   ```text
   b = 1 + sqrt(2)
   k = floor(1 / (b * epsilon^3)) + 1
   ```

   Smaller epsilon gives a larger `k`, which generally increases the copied graph size and runtime.

4. **Case 1 estimator**

   Case 1 starts from the unmatched induced graph \(G[V \setminus V(M)]\). It estimates an inner matching \(M'\) using the random greedy maximal matching oracle. Then it builds a lazy copied-graph view, `Case1CopiedView`, that represents the theoretical graph \(G_1'\) without constructing all copied edges.

   It estimates:

   - `Mprime_estimate`: size of the inner matching \(M'\),
   - `B1_estimate`: matching size in the Case 1 copied graph.

   The final Case 1 value is:

   ```text
   mu1 = |M| + (1 - 1/b) * Mprime_estimate + (1 / (k*b)) * B1_estimate
   ```

5. **Case 2 estimator**

   Case 2 builds another lazy copied graph, `Case2CopiedView`. Vertices already matched by `M` receive `k` copies, and unmatched vertices receive `ceil(k*b)` copies. The view exposes copied degrees and copied neighbors on demand.

   It estimates:

   - `B2_estimate`: matching size in the Case 2 copied graph.

   The final Case 2 value is:

   ```text
   mu2 = (1 - 1/b) * |M| + (1 / (k*b)) * B2_estimate
   ```

6. **Final estimate**

   The modern estimator returns:

   ```text
   estimate = max(mu1, mu2)
   ```

   The benchmark records which case was selected for every trial.

## Local Random Greedy Matching Oracle

The file `modernAlgo/behnezhad_rgmm_oracle.py` implements a local oracle for random greedy maximal matching. Instead of assigning and sorting ranks for all edges globally, it lazily exposes rank intervals only as needed.

Important methods:

- `vertex_matched(v)`: returns whether a vertex is matched in the simulated random greedy matching.
- `matched_edge(v)`: returns the edge incident to `v` in that matching, if one exists.
- `matched_vertices_fraction(sample_vertices)`: estimates a matching size by sampling vertices and measuring what fraction are matched.

This oracle is what makes the copied-graph cases practical: the algorithm can ask local questions about a huge implicit graph without explicitly storing all copied vertices and copied edges.

## Graph Families

The benchmark uses three graph generators:

- **ER (`er`)**: balanced bipartite Erdos-Renyi graph with edge probability `p = expected_degree / n`.
- **Power-law (`powerlaw`)**: weighted endpoint sampling creates skewed degree distributions while matching the approximate ER edge count.
- **Near-regular (`sqrtreg`)**: each left vertex chooses about `ceil(sqrt(n))` right neighbors.

These families are intended to test the estimator under uniform random structure, skewed degree structure, and dense-ish regular structure.

## Running the Project

Install the Python dependencies:

```bash
python -m pip install networkx matplotlib
```

The commands below use `python`. On Windows, `py -3` can be substituted if the Python launcher is configured instead.

Run the small console experiment:

```bash
python experiment.py
```

Run the main benchmark:

```bash
python benchmark.py --tier tier1 --progress
```

Run the epsilon sensitivity sweep:

```bash
python benchmark.py --tier tier2 --progress
```

Generate summary CSVs:

```bash
python summarize_benchmark.py --input results/tier1_main.csv --output-dir results/tier1_summaries
python summarize_benchmark.py --input results/tier2_epsilon_sweep.csv --output-dir results/tier2_summaries
```

Generate plots:

```bash
python plot_benchmark.py --input results/tier1_main.csv --output-dir figures/tier1 --mode full
python plot_benchmark.py --input results/tier2_epsilon_sweep.csv --output-dir figures/tier2 --mode epsilon-only
```

To override the theorem-derived capacity parameter:

```bash
python benchmark.py --tier custom --n-values 50,100 --expected-degrees 4,8 --epsilons 0.3 --k 8 --progress
```

## Existing Results

The repository already contains generated benchmark outputs in `results/` and generated figures in `figures/`.

For the Tier 1 benchmark, the summary by graph family reports:

| Graph family | Trials | Mean OPT | Mean greedy ratio | Mean randomized greedy ratio | Mean modern ratio | Case 1 selected | Case 2 selected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ER | 500 | 361.666 | 0.9198 | 0.8987 | 0.9195 | 100.0% | 0.0% |
| Power-law | 500 | 172.118 | 0.8072 | 0.8047 | 0.7853 | 73.8% | 26.2% |
| Near-regular sqrt(n) | 100 | 380.000 | 0.9465 | 0.9481 | 0.9556 | 100.0% | 0.0% |

These values are stored in `results/tier1_summaries/summary_by_family.csv`.

## Benchmark Columns

`benchmark.py` writes one row per trial. Key columns include:

- `optimal_size`: exact maximum matching size from Hopcroft-Karp.
- `greedy_size`, `random_greedy_size`: baseline matching sizes.
- `modern_estimate`: final modern estimate `max(mu1, mu2)`.
- `modern_mu1`, `modern_mu2`: the two modern case estimates.
- `selected_case`: which modern case produced the final estimate.
- `M_size`, `Mprime_estimate`, `B1_estimate`, `B2_estimate`: internal modern decomposition.
- `epsilon`, `k`, `sparsify_c`: modern algorithm parameters.
- `modern_ratio`: `modern_estimate / optimal_size`.
- `t_modern_sparsify`, `t_modern_case1`, `t_modern_case2`: timing breakdown for the modern pipeline.

## Notes for Graders

The main code to inspect is `modernAlgo/modern_algo.py`, which wires together the full modern algorithm. The most important supporting files are:

- `modernAlgo/sparsify.py` for the initial matching `M`,
- `modernAlgo/behnezhad_rgmm_oracle.py` for the local RGMM oracle,
- `modernAlgo/case1/estimator.py` and `modernAlgo/case1/view_copied.py` for Case 1,
- `modernAlgo/case2/estimator.py` and `modernAlgo/case2/copied_graph.py` for Case 2.

The implementation is intentionally oracle-oriented. The copied graphs can be much larger than the original graph, especially when epsilon is small and `k` is large, so the project represents them as lazy graph views. This is the core engineering decision behind the modern algorithm implementation.
