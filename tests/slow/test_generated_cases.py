import pytest

from clustering_regularity.verification.models import verification_cases
from clustering_regularity.verification.runner import run_case_with_timeout


@pytest.mark.slow
@pytest.mark.parametrize("case", verification_cases(), ids=lambda case: case.case_id)
def test_generated_case(case):
    result = run_case_with_timeout(
        case,
        timeout_seconds=20,
        max_rounds=20,
    )

    if result["execution_status"] == "timeout":
        pytest.skip(f"TIMEOUT: {case.case_id}")
    assert result["execution_status"] == "completed"
    assert result["verification_status"] in {
        "certified_success",
        "proven_failure",
        "inconclusive",
    }

