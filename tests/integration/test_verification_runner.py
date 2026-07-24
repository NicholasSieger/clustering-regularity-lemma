import pytest

from clustering_regularity.verification.models import (
    VerificationCase,
    VerificationConfig,
)
from clustering_regularity.verification.runner import run_case_with_timeout


@pytest.mark.integration
def test_completed_case_has_compact_algorithm_summary():
    result = run_case_with_timeout(
        VerificationCase("sbm", 30, 0.05, 20260709),
        timeout_seconds=20,
        max_rounds=20,
    )

    assert result["execution_status"] == "completed"
    assert "directions_considered" not in result
    assert "accepted_directions" not in result
    assert result["algorithm"]["round_count"] >= 1


@pytest.mark.integration
def test_timeout_is_separate_from_verification():
    result = run_case_with_timeout(
        VerificationCase("sbm", 30, 0.15, 1234),
        timeout_seconds=0.001,
    )

    assert result["execution_status"] == "timeout"
    assert result["verification_status"] is None

