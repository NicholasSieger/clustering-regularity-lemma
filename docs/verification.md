# Verification

The verifier reports two independent views.

`paper_partition_status` reproduces the paper's operational low-gamma,
triangle-degree, and deviation checks. `verification_status` evaluates the
literal universal definition and has three states:

- `certified_success`: exhaustive enumeration or a sound spectral bound
  certifies enough pathweight.
- `proven_failure`: replayable witnesses prove at least epsilon pathweight is
  irregular.
- `inconclusive`: neither condition is established.

Small cells are enumerated exactly. Larger cells use an inflated spectral norm
bound and deterministic witness search. Search can prove failure but never
certify success.

Reports are compact by default. They include aggregate counts and at most
`max_report_cells` prioritized cells. Zero-path Cartesian pairs are included in
the cell counts without emitting one JSON object per empty pair.

