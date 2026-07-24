import numpy as np

from clustering_regularity.core.models import RefinementCut
from clustering_regularity.storage.local import LocalPartitionStore


def test_partition_generations_are_disjoint_integer_labels(tmp_path):
    store = LocalPartitionStore(tmp_path / "generations", edge_count=6)
    generation = store.commit_refinement(
        0,
        [
            RefinementCut(0, (0, 1, 2)),
            RefinementCut(0, (0, 3)),
        ],
        [RefinementCut(0, (1, 3, 5))],
    )

    labels_a = store.labels(generation, "A")
    labels_b = store.labels(generation, "B")
    try:
        assert store.part_count(generation, "A") == 4
        assert store.part_count(generation, "B") == 2
        assert len(labels_a) == len(labels_b) == 6
        assert set(labels_a.tolist()) == set(range(4))
        assert set(labels_b.tolist()) == {0, 1}
    finally:
        labels_a._mmap.close()
        labels_b._mmap.close()


def test_empty_edge_generation(tmp_path):
    store = LocalPartitionStore(tmp_path / "generations", edge_count=0)
    labels = store.labels(0, "A")
    try:
        assert labels.shape == (0,)
        assert store.part_count(0, "A") == 0
    finally:
        labels._mmap.close()


def test_refinement_cut_can_stream_indices_from_disk(tmp_path):
    store = LocalPartitionStore(tmp_path / "generations", edge_count=6)
    index_path = tmp_path / "selected.indices"
    np.asarray([1, 3, 5], dtype=np.int64).tofile(index_path)

    generation = store.commit_refinement(
        0,
        [RefinementCut(0, edge_indices_path=index_path)],
        [],
    )
    labels = store.labels(generation, "A")
    try:
        assert labels.tolist() == [0, 1, 0, 1, 0, 1]
    finally:
        labels._mmap.close()
