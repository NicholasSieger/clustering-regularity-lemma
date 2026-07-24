# Data Flow

1. The graph index groups oriented edges and one sparse closure matrix by
   middle vertex. It does not construct the tripartite cover.
2. A partition generation stores one E12 and one E23 integer label vector.
3. The first pass of a round loads each middle shard once and reduces
   pathweights and triangle counts by cell.
4. The second pass loads each shard once and reduces local irregularity,
   deviation, and branch weights using the globally computed cell gamma.
5. If refinement is required, a third shard pass writes selected edge indices
   to sparse work files. No round-wide witness masks are retained in memory.
6. The sparse cuts are committed as a new immutable partition generation.
7. The engine emits one typed summary per round. Observers decide whether to
   print or persist it.

The two reduction passes are associative by cell identifier. The cut pass
emits append-only records keyed by cell. A future distributed executor can map
middle shards to workers and reduce or merge the same records without changing
the mathematical core.
