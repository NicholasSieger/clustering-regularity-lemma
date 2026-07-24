import numpy as np
import pytest
from scipy import sparse

from clustering_regularity.config import PaperParameters
from clustering_regularity.core.local_checks import LocalCheck


def test_zero_edge_local_check():
    check = LocalCheck(sparse.csr_matrix((3, 4)), PaperParameters())
    positive, positive_count, negative, negative_count = check.irregular_edges(0.0)

    assert check.pathweight == 12
    assert check.triangle_count == 0
    assert check.deviation(0.0) == 0.0
    assert positive_count == negative_count == 0
    assert not positive.any()
    assert not negative.any()


def test_sparse_deviation_matches_dense_definition():
    matrix = np.asarray(
        [
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [1, 1, 0, 1],
        ],
        dtype=float,
    )
    gamma = 0.4
    check = LocalCheck(sparse.csr_matrix(matrix), PaperParameters())
    expected = np.sum(matrix @ matrix.T - gamma**2 * matrix.shape[1])

    assert check.deviation(gamma) == pytest.approx(expected)


def test_deviation_split_has_stable_dimensions():
    matrix = sparse.csr_matrix(np.eye(4))
    check = LocalCheck(matrix, PaperParameters())

    split_a, split_b = check.deviation_split(0.25)

    assert split_a.shape == (4,)
    assert split_b.shape == (4,)
    assert split_a.dtype == split_b.dtype == bool

