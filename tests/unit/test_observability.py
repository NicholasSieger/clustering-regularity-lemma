import json

from clustering_regularity.core.models import RoundSummary
from clustering_regularity.observability.events import RoundCompleted
from clustering_regularity.observability.observers import JsonlObserver


def test_jsonl_observer_streams_one_bounded_round_record(tmp_path):
    path = tmp_path / "events.jsonl"
    observer = JsonlObserver(path)
    observer.publish(
        RoundCompleted(
            RoundSummary(
                round_index=0,
                generation=0,
                part_count_a=1,
                part_count_b=1,
                evaluated_nonzero_cells=1,
                total_cell_count=1,
                failed_cell_count=0,
                total_pathweight=10,
                failed_pathweight=0,
                branch_counts={"regular": 1},
                elapsed_seconds=0.1,
            )
        )
    )

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["event"] == "round_completed"
    assert "cells" not in records[0]

