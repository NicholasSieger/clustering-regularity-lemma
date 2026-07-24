"""Paper-faithful and literal-definition verification."""

from .cells import (
    finite_scale_threshold_report,
    make_cell_group,
    validate_witness,
)
from .mathematics import (
    aggregate_definition_status,
    compute_partition_regularity_report,
    verify_cell,
)
from .models import VerificationCase, VerificationConfig

__all__ = [
    "VerificationCase",
    "VerificationConfig",
    "aggregate_definition_status",
    "compute_partition_regularity_report",
    "finite_scale_threshold_report",
    "make_cell_group",
    "validate_witness",
    "verify_cell",
]
