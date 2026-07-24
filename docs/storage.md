# Storage

`index/` contains immutable middle-vertex shards. Each shard has JSON node and
edge-range metadata plus a compressed SciPy CSR closure matrix.

`generations/NNNNNNNN/` contains memory-mappable E12 and E23 label vectors and
a compact manifest. `CURRENT` is updated atomically only after both vectors and
the manifest are complete.

`logs/` contains optional round-level JSONL. Detailed verifier artifacts are
separate from algorithm state.

`work/` contains short-lived sparse refinement-cut files. The engine removes a
round's work files after the generation commit succeeds or fails.

The local backend implements the same boundaries expected from object-store or
distributed backends: immutable inputs, deterministic shard identifiers,
idempotent work units, reduction by cell identifier, and coordinator-owned
generation commits.
