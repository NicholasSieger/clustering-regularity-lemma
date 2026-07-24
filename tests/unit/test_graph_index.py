import networkx as nx
import numpy as np
import pytest

from clustering_regularity.graph import GraphIndex


def test_index_uses_oriented_edges_without_tripartite_materialization(tmp_path):
    graph = nx.path_graph(4)
    index = GraphIndex.build(graph, tmp_path / "index")

    assert index.node_count == 4
    assert index.edge_count == 6
    assert len(list(index.iter_edge_records("A"))) == 6
    assert len(list(index.iter_edge_records("B"))) == 6


def test_complete_graph_link_shard_is_neighbor_adjacency(tmp_path):
    graph = nx.complete_graph(4)
    index = GraphIndex.build(graph, tmp_path / "index")
    shard = next(
        index.load_shard(shard_id)
        for shard_id in index.iter_shard_ids()
        if index.load_shard(shard_id).middle == 0
    )

    assert shard.closure.shape == (3, 3)
    assert shard.closure.nnz == 6
    assert np.all(shard.closure.diagonal() == 0)


def test_tuple_nodes_round_trip(tmp_path):
    graph = nx.Graph()
    graph.add_edge(("left", 1), ("right", 2))
    index = GraphIndex.build(graph, tmp_path / "index")

    nodes = {index.load_shard(item).middle for item in index.iter_shard_ids()}
    assert nodes == {("left", 1), ("right", 2)}


@pytest.mark.parametrize(
    "graph,error",
    [
        (nx.DiGraph([(0, 1)]), TypeError),
        (nx.MultiGraph([(0, 1), (0, 1)]), TypeError),
        (nx.Graph([(0, 0)]), ValueError),
    ],
)
def test_index_rejects_graphs_outside_the_paper_model(tmp_path, graph, error):
    with pytest.raises(error):
        GraphIndex.build(graph, tmp_path / "index")
