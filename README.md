# Clustering Regularity

This package implements the algorithmic clustering regularity lemma using
file-backed link-graph shards and edge-partition generations. The core engine
loads one middle-vertex link graph at a time and does not materialize the
tripartite cover.

## Install

Python 3.9 or newer is required.

```powershell
python -m pip install -e ".[test]"
```

## Run

```powershell
clustering-regularity run graph.edgelist --workspace runs/example --epsilon 0.05
```

The workspace contains an immutable graph index, partition generations, a
compact `result.json`, and optional round-level JSONL events. Use
`--max-rounds` to bound exploratory runs.

```powershell
clustering-regularity verify graph.edgelist --workspace runs/example
```

## Python API

```python
import networkx as nx
from clustering_regularity import run_graph

graph = nx.read_edgelist("graph.edgelist", nodetype=int)
result = run_graph(graph, "runs/example", epsilon=0.05)
labels_a = result.labels("A")
labels_b = result.labels("B")
```

## Layout

```text
src/clustering_regularity/
  core/           paper thresholds, local checks, and refinement decisions
  graph/          immutable, middle-vertex link-graph index
  storage/        partition-store protocol and local generation backend
  observability/  typed events and optional console/JSONL observers
  verification/   paper checks, literal certificates, generators, and reports
  api.py           Python entry point
  cli.py           command-line entry point
  engine.py        storage-backed round coordinator
tests/
  unit/            mathematical and storage contracts
  integration/     end-to-end engine and worker behavior
  slow/            27 generated graph cases, excluded from default pytest
docs/              algorithm mapping, data flow, storage, and verification
```

See `docs/algorithm.md`, `docs/data-flow.md`, and `docs/storage.md` for the
paper mapping and execution model.
