import json

from clustering_regularity.verification.models import REPORT_SCHEMA_VERSION
from clustering_regularity.verification.report import (
    report_document,
    write_verification_report,
)


def test_report_schema_sorts_cases():
    document = report_document(
        [{"case_id": "z"}, {"case_id": "a"}],
        {"epsilon": 0.05},
    )

    assert document["schema_version"] == REPORT_SCHEMA_VERSION
    assert document["run_config"] == {"epsilon": 0.05}
    assert [case["case_id"] for case in document["cases"]] == ["a", "z"]


def test_report_write_atomically_replaces_existing_file(tmp_path):
    output = tmp_path / "report.json"
    output.write_text("obsolete", encoding="utf-8")

    write_verification_report([{"case_id": "case"}], output)

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["cases"] == [{"case_id": "case"}]
    assert list(tmp_path.glob(".report_*.json")) == []
