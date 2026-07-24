import networkx as nx
import pytest

from clustering_regularity.verification.generators import (
    algorithm_clustering_coefficient,
    generate_graph,
    generate_sbm_near_epsilon_graph,
)


@pytest.mark.parametrize("family", ("sbm", "powerlaw_line", "watts_strogatz"))
@pytest.mark.parametrize("density", (0.05, 0.15, 0.35))
def test_generators_control_n30_density(family, density):
    graph, _ = generate_graph(family, 30, density, 1234)

    assert graph.number_of_nodes() == 30
    assert abs(nx.density(graph) - density) <= 0.02


def test_near_epsilon_generator():
    graph, metadata = generate_sbm_near_epsilon_graph(60, 0.06, 20260904)

    assert abs(algorithm_clustering_coefficient(graph) - 0.05) < 0.001
    assert metadata["selected_root_gamma"] >= 0.05

