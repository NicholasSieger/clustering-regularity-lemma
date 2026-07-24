# Algorithm and Paper Mapping

The engine implements the paper thresholds through `PaperParameters`:

| Paper symbol | Code property | Use |
|---|---|---|
| epsilon | `epsilon` | low-gamma and partition stopping threshold |
| delta 1 | `delta_1` | local deviation threshold |
| delta 2 | `delta_2` | aggregate deviation threshold |
| delta 3 | `delta_3` | triangle-degree tolerance |
| delta 4 | `delta_4` | irregular-edge count threshold |
| delta 5 | `delta_5` | aggregate irregularity threshold |
| delta 6 | `delta_6` | deviation witness split |

For every Cartesian pair of current E12 and E23 classes, the engine computes
pathweight and triangle count exactly. Cells with gamma below epsilon pass the
paper's low-gamma rule. Remaining cells use the per-middle triangle-degree and
deviation checks. All witness cuts from a round are applied as a common
refinement. The algorithm stops when failed-cell pathweight is strictly less
than epsilon times total pathweight.

Zero-path cells are retained logically and pass vacuously. Edge labels are
always disjoint and exhaustive because each oriented edge has exactly one
integer label in each generation.

