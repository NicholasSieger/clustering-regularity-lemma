import numpy as np
import networkx as nx
import pytest

from clustering_task import Task


def make_nodes(count, part):
    return [(idx, part) for idx in range(count)]


def make_complete_link_graph(a_count, b_count):
    A = make_nodes(a_count, 0)
    B = make_nodes(b_count, 2)
    G = nx.Graph()
    G.add_nodes_from(A, part=0)
    G.add_nodes_from(B, part=2)
    G.add_edges_from((a, b) for a in A for b in B)
    return G, A, B


def make_empty_link_graph(a_count, b_count):
    A = make_nodes(a_count, 0)
    B = make_nodes(b_count, 2)
    G = nx.Graph()
    G.add_nodes_from(A, part=0)
    G.add_nodes_from(B, part=2)
    return G, A, B


def make_random_link_graph(a_count, b_count, probability, seed):
    rng = np.random.default_rng(seed)
    A = make_nodes(a_count, 0)
    B = make_nodes(b_count, 2)
    G = nx.Graph()
    G.add_nodes_from(A, part=0)
    G.add_nodes_from(B, part=2)
    for a in A:
        for b in B:
            if rng.random() < probability:
                G.add_edge(a, b)
    return G, A, B


def apply_mask(nodes, mask):
    return [node for node, include in zip(nodes, mask) if include]


def expected_irregular_vertices(link_graph, A, B, gamma, delta):
    positive, negative = expected_signed_irregular_vertices(link_graph, A, B, gamma, delta)
    return positive | negative


def expected_signed_irregular_vertices(link_graph, A, B, gamma, delta):
    b_set = set(B)
    deg_B_v = len(B)
    expected_degree = gamma * deg_B_v
    threshold = delta * deg_B_v

    positive = []
    negative = []
    for u in A:
        observed_degree = sum(1 for neighbor in link_graph.neighbors(u) if neighbor in b_set)
        difference = observed_degree - expected_degree
        positive.append(difference > threshold)
        negative.append(difference < -threshold)
    return np.array(positive, dtype=bool), np.array(negative, dtype=bool)


def common_neighbor_count(link_graph, u, u_prime, B):
    b_set = set(B)
    return sum(
        1
        for neighbor in link_graph.neighbors(u)
        if neighbor in b_set and link_graph.has_edge(u_prime, neighbor)
    )


def expected_common_neighbor_matrix(link_graph, A, B):
    return np.array(
        [
            [common_neighbor_count(link_graph, u, u_prime, B) for u_prime in A]
            for u in A
        ],
        dtype=float,
    )


def expected_local_deviation(link_graph, A, B, gamma):
    common_neighbors = expected_common_neighbor_matrix(link_graph, A, B)
    return np.sum(common_neighbors - (gamma**2) * len(B))


def expected_split_masks(link_graph, A, B, gamma, delta, delta_2):
    deg_B_v = len(B)
    expected_degree = gamma * deg_B_v
    candidate_mask = np.array(
        [
            abs(sum(1 for neighbor in link_graph.neighbors(u) if neighbor in set(B)) - expected_degree)
            < delta * deg_B_v
            for u in A
        ],
        dtype=bool,
    )
    common_neighbors = expected_common_neighbor_matrix(link_graph, A, B)
    deviation_matrix = common_neighbors - (gamma**2) * deg_B_v
    scores = np.sum(deviation_matrix, axis=1)
    candidate_indices = np.where(candidate_mask)[0]
    if candidate_indices.size == 0:
        candidate_indices = np.arange(len(A))

    u_star_index = candidate_indices[np.argmax(scores[candidate_indices])]
    u_star = A[u_star_index]
    A_prime = common_neighbors[u_star_index, :] > delta_2 * deg_B_v
    B_prime = np.array([link_graph.has_edge(u_star, b) for b in B], dtype=bool)
    return A_prime, B_prime, u_star_index


def assert_irregular_vertices_match(link_graph, A, B, gamma, delta):
    task = Task(link_graph, (A, B), eps=0.1, irreg_vtx_threshold=delta)
    actual_positive, actual_positive_count, actual_negative, actual_negative_count = task.compute_irregular_vertices(gamma)
    expected_positive, expected_negative = expected_signed_irregular_vertices(link_graph, A, B, gamma, delta)

    assert np.array_equal(actual_positive, expected_positive)
    assert actual_positive_count == np.sum(expected_positive)
    assert np.array_equal(actual_negative, expected_negative)
    assert actual_negative_count == np.sum(expected_negative)


def assert_local_deviation_matches(link_graph, A, B, gamma):
    task = Task(link_graph, (A, B), eps=0.1)
    actual = task.compute_local_deviation(gamma)
    expected = expected_local_deviation(link_graph, A, B, gamma)

    assert actual == pytest.approx(expected)
    assert np.array_equal(
        task.common_neighbor_matrix,
        expected_common_neighbor_matrix(link_graph, A, B),
    )


def assert_split_masks_match(link_graph, A, B, gamma, delta, delta_2):
    task = Task(
        link_graph,
        (A, B),
        eps=0.1,
        dev_vtx_threshold=delta,
        dev_split_threshold=delta_2,
    )
    actual_A_prime, actual_B_prime = task.produce_new_masks(gamma)
    expected_A_prime, expected_B_prime, _ = expected_split_masks(
        link_graph, A, B, gamma, delta, delta_2
    )

    assert np.array_equal(actual_A_prime, expected_A_prime)
    assert np.array_equal(actual_B_prime, expected_B_prime)


@pytest.mark.parametrize(
    ("delta", "expected_count"),
    [
        (0.49, 5),
        (0.50, 0),
        (0.75, 0),
    ],
)
def test_irregular_vertices_on_complete_link_graph_for_delta_values(delta, expected_count):
    link_graph, A, B = make_complete_link_graph(a_count=5, b_count=4)

    assert_irregular_vertices_match(link_graph, A, B, gamma=0.5, delta=delta)
    task = Task(link_graph, (A, B), eps=0.1, irreg_vtx_threshold=delta)
    _, actual_count, _, negative_count = task.compute_irregular_vertices(gamma=0.5)

    assert actual_count == expected_count
    assert negative_count == 0


@pytest.mark.parametrize(
    ("delta", "expected_count"),
    [
        (0.49, 5),
        (0.50, 0),
        (0.75, 0),
    ],
)
def test_irregular_vertices_on_empty_link_graph_for_delta_values(delta, expected_count):
    link_graph, A, B = make_empty_link_graph(a_count=5, b_count=4)

    assert_irregular_vertices_match(link_graph, A, B, gamma=0.5, delta=delta)
    task = Task(link_graph, (A, B), eps=0.1, irreg_vtx_threshold=delta)
    positive_mask, positive_count, negative_mask, negative_count = task.compute_irregular_vertices(gamma=0.5)

    assert positive_count == 0
    assert not np.any(positive_mask)
    assert negative_count == expected_count
    assert np.array_equal(negative_mask, expected_irregular_vertices(link_graph, A, B, 0.5, delta))


@pytest.mark.parametrize(
    ("delta", "expected_mask"),
    [
        (0.10, np.array([True, True, True])),
        (0.20, np.array([False, False, True])),
        (0.50, np.array([False, False, False])),
    ],
)
def test_irregular_vertices_respect_A_and_B_masks(delta, expected_mask):
    link_graph, A_full, B_full = make_empty_link_graph(a_count=4, b_count=5)
    edges = [
        (A_full[0], B_full[0]),
        (A_full[0], B_full[1]),
        (A_full[0], B_full[2]),
        (A_full[1], B_full[0]),
        (A_full[2], B_full[2]),
        (A_full[2], B_full[3]),
        (A_full[2], B_full[4]),
    ]
    link_graph.add_edges_from(edges)

    A = apply_mask(A_full, [True, True, False, True])
    B = apply_mask(B_full, [True, True, False, False, True])

    task = Task(link_graph, (A, B), eps=0.1, irreg_vtx_threshold=delta)
    actual_positive, actual_positive_count, actual_negative, actual_negative_count = task.compute_irregular_vertices(gamma=0.5)
    actual_mask = actual_positive | actual_negative

    assert np.array_equal(actual_mask, expected_mask)
    assert actual_positive_count + actual_negative_count == np.sum(expected_mask)
    assert np.array_equal(actual_mask, expected_irregular_vertices(link_graph, A, B, 0.5, delta))


@pytest.mark.parametrize("delta", [0.0, 0.15, 0.4, 0.8])
def test_irregular_vertices_on_random_link_graph_with_masks(delta):
    link_graph, A_full, B_full = make_random_link_graph(
        a_count=7,
        b_count=6,
        probability=0.35,
        seed=20260514,
    )
    A = apply_mask(A_full, [True, False, True, True, False, True, True])
    B = apply_mask(B_full, [False, True, True, False, True, True])

    assert_irregular_vertices_match(link_graph, A, B, gamma=0.4, delta=delta)


def test_irregular_vertices_use_constructor_threshold_not_eps_default():
    link_graph, A, B = make_complete_link_graph(a_count=3, b_count=4)

    small_delta_task = Task(link_graph, (A, B), eps=0.5, irreg_vtx_threshold=0.49)
    large_delta_task = Task(link_graph, (A, B), eps=0.5, irreg_vtx_threshold=0.50)

    _, small_delta_count, _, small_delta_negative_count = small_delta_task.compute_irregular_vertices(gamma=0.5)
    _, large_delta_count, _, large_delta_negative_count = large_delta_task.compute_irregular_vertices(gamma=0.5)

    assert small_delta_count == len(A)
    assert small_delta_negative_count == 0
    assert large_delta_count == 0
    assert large_delta_negative_count == 0


@pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0])
def test_local_deviation_on_complete_link_graph(gamma):
    link_graph, A, B = make_complete_link_graph(a_count=4, b_count=5)

    assert_local_deviation_matches(link_graph, A, B, gamma)


@pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0])
def test_local_deviation_on_empty_link_graph(gamma):
    link_graph, A, B = make_empty_link_graph(a_count=4, b_count=5)

    assert_local_deviation_matches(link_graph, A, B, gamma)


@pytest.mark.parametrize("gamma", [0.25, 0.5, 0.75])
def test_local_deviation_respects_A_and_B_masks(gamma):
    link_graph, A_full, B_full = make_empty_link_graph(a_count=5, b_count=6)
    link_graph.add_edges_from(
        [
            (A_full[0], B_full[0]),
            (A_full[0], B_full[1]),
            (A_full[1], B_full[1]),
            (A_full[1], B_full[2]),
            (A_full[2], B_full[2]),
            (A_full[3], B_full[0]),
            (A_full[3], B_full[3]),
            (A_full[4], B_full[4]),
            (A_full[4], B_full[5]),
        ]
    )
    A = apply_mask(A_full, [True, False, True, True, False])
    B = apply_mask(B_full, [True, True, False, True, False, True])

    assert_local_deviation_matches(link_graph, A, B, gamma)


@pytest.mark.parametrize("gamma", [0.2, 0.4, 0.7])
def test_local_deviation_on_random_link_graph_with_masks(gamma):
    link_graph, A_full, B_full = make_random_link_graph(
        a_count=8,
        b_count=7,
        probability=0.4,
        seed=20260515,
    )
    A = apply_mask(A_full, [True, True, False, True, False, True, True, False])
    B = apply_mask(B_full, [True, False, True, True, False, True, True])

    assert_local_deviation_matches(link_graph, A, B, gamma)


def test_split_masks_on_complete_link_graph():
    link_graph, A, B = make_complete_link_graph(a_count=4, b_count=5)

    assert_split_masks_match(link_graph, A, B, gamma=1.0, delta=0.1, delta_2=0.2)
    A_prime, B_prime = Task(
        link_graph,
        (A, B),
        eps=0.1,
        dev_vtx_threshold=0.1,
        dev_split_threshold=0.2,
    ).produce_new_masks(gamma=1.0)

    assert np.array_equal(A_prime, np.ones(len(A), dtype=bool))
    assert np.array_equal(B_prime, np.ones(len(B), dtype=bool))


def test_split_masks_on_empty_link_graph():
    link_graph, A, B = make_empty_link_graph(a_count=4, b_count=5)

    assert_split_masks_match(link_graph, A, B, gamma=0.0, delta=0.1, delta_2=0.2)
    A_prime, B_prime = Task(
        link_graph,
        (A, B),
        eps=0.1,
        dev_vtx_threshold=0.1,
        dev_split_threshold=0.2,
    ).produce_new_masks(gamma=0.0)

    assert np.array_equal(A_prime, np.zeros(len(A), dtype=bool))
    assert np.array_equal(B_prime, np.zeros(len(B), dtype=bool))


@pytest.mark.parametrize(
    ("delta", "delta_2", "expected_A_prime", "expected_B_prime"),
    [
        (0.26, 0.49, np.array([True, True, False]), np.array([True, True, True, False])),
        (0.26, 0.75, np.array([False, False, False]), np.array([True, True, True, False])),
        (0.10, 0.49, np.array([True, True, False]), np.array([True, True, False, False])),
    ],
)
def test_split_masks_respect_A_and_B_masks(delta, delta_2, expected_A_prime, expected_B_prime):
    link_graph, A_full, B_full = make_empty_link_graph(a_count=4, b_count=5)
    link_graph.add_edges_from(
        [
            (A_full[0], B_full[0]),
            (A_full[0], B_full[1]),
            (A_full[0], B_full[4]),
            (A_full[1], B_full[0]),
            (A_full[1], B_full[1]),
            (A_full[1], B_full[2]),
            (A_full[2], B_full[3]),
            (A_full[3], B_full[0]),
            (A_full[3], B_full[2]),
        ]
    )
    A = apply_mask(A_full, [True, True, True, False])
    B = apply_mask(B_full, [True, True, True, True, False])

    task = Task(
        link_graph,
        (A, B),
        eps=0.1,
        dev_vtx_threshold=delta,
        dev_split_threshold=delta_2,
    )
    actual_A_prime, actual_B_prime = task.produce_new_masks(gamma=0.5)

    assert np.array_equal(actual_A_prime, expected_A_prime)
    assert np.array_equal(actual_B_prime, expected_B_prime)
    assert_split_masks_match(link_graph, A, B, gamma=0.5, delta=delta, delta_2=delta_2)


@pytest.mark.parametrize(
    ("delta", "delta_2"),
    [
        (0.15, 0.0),
        (0.35, 0.25),
        (0.75, 0.5),
    ],
)
def test_split_masks_on_random_link_graph_with_masks(delta, delta_2):
    link_graph, A_full, B_full = make_random_link_graph(
        a_count=7,
        b_count=6,
        probability=0.45,
        seed=20260516,
    )
    A = apply_mask(A_full, [True, False, True, True, True, False, True])
    B = apply_mask(B_full, [True, True, False, True, False, True])

    assert_split_masks_match(link_graph, A, B, gamma=0.5, delta=delta, delta_2=delta_2)
