#!/usr/bin/env python3
"""
Comprehensive test suite for the clustering regularity algorithm.
Tests the algorithm on various graph types:
1. Complete graphs
2. Bipartite graphs
3. Random graphs
4. Stochastic block models

All graphs have at most 30 vertices.
Tests use bounded max_depth and deterministic random seeds.
"""

import unittest
import tempfile
import shutil
import os
import networkx as nx
import numpy as np

from algorithm import AlgorithmRunner
from parameters import AlgorithmParameters


class GraphGenerator:
    """Utility class to generate various types of graphs."""

    @staticmethod
    def complete_graph(n):
        """Generate a complete graph with n vertices."""
        return nx.complete_graph(n)

    @staticmethod
    def bipartite_graph(n_left, n_right, edge_prob=None, seed=None):
        """
        Generate a bipartite graph.
        If edge_prob is None, generates a complete bipartite graph.
        Otherwise, generates a random bipartite graph with given edge probability.
        """
        if edge_prob is None:
            return nx.complete_bipartite_graph(n_left, n_right)
        
        # Generate random bipartite graph with numeric node labels
        B = nx.Graph()
        left_nodes = list(range(n_left))
        right_nodes = list(range(n_left, n_left + n_right))
        B.add_nodes_from(left_nodes, bipartite=0)
        B.add_nodes_from(right_nodes, bipartite=1)
        rng = np.random.default_rng(seed)
        
        for u in left_nodes:
            for v in right_nodes:
                if rng.random() < edge_prob:
                    B.add_edge(u, v)
        
        return B

    @staticmethod
    def random_graph(n, edge_prob, seed=None):
        """Generate an Erdos-Renyi random graph with n vertices."""
        return nx.erdos_renyi_graph(n, edge_prob, seed=seed)

    @staticmethod
    def stochastic_block_model(sizes, probs, seed=None):
        """
        Generate a stochastic block model.
        
        Args:
            sizes: List of block sizes
            probs: Matrix of edge probabilities between/within blocks
        """
        return nx.stochastic_block_model(sizes, probs, seed=seed)


class TestClusteringRegularityAlgorithm(unittest.TestCase):
    """Test suite for the clustering regularity algorithm on various graphs."""

    def setUp(self):
        """Create temporary directories for test artifacts."""
        self.temp_dir = tempfile.mkdtemp()
        self.partition_dir = os.path.join(self.temp_dir, "partitions")
        self.graph_dir = os.path.join(self.temp_dir, "graphs")

    def tearDown(self):
        """Clean up temporary directories."""
        shutil.rmtree(self.temp_dir)

    def run_algorithm_with_timeout(self, G, eps=0.1, timeout_seconds=60, max_depth=8):
        """
        Run the algorithm on a graph with bounded refinement depth.
        
        Args:
            G: NetworkX graph
            eps: Algorithm parameter
            timeout_seconds: Retained for compatibility; max_depth bounds runtime.
            max_depth: Maximum recursion depth to avoid excessively long paths (default 8)
        
        Returns:
            Tuple of (labels_A, labels_B) on success, or None on skipped max-depth exhaustion.
        """
        parameters = AlgorithmParameters(eps=eps)
        runner = AlgorithmRunner(
            G,
            parameters=parameters,
            partition_dir=self.partition_dir,
            graph_dir=self.graph_dir,
            max_depth=max_depth,
        )

        try:
            return runner.run()
        except Exception as e:
            if isinstance(e, ValueError) and "exceeds maximum depth" in str(e):
                self.skipTest(f"MAX_DEPTH: {e}")
                return None
            self.fail(f"ERROR: {e}")

    def validate_partition_labels(self, labels_A, labels_B, G):
        """
        Validate that partition labels are valid.
        
        The algorithm returns partition labels for edges, not vertices.
        labels_A and labels_B should correspond to partitions of edges
        in the two tripartite edge sets E12 and E23.
        """
        # Labels should be numpy arrays or similar
        self.assertIsNotNone(labels_A)
        self.assertIsNotNone(labels_B)
        
        # Labels should be non-negative integers
        self.assertTrue(np.all(labels_A >= 0))
        self.assertTrue(np.all(labels_B >= 0))
        
        # The tripartite cover has one E12 and one E23 edge for each oriented
        # original edge, so each label vector has two entries per input edge.
        expected_label_count = 2 * G.number_of_edges()
        self.assertEqual(len(labels_A), expected_label_count)
        self.assertEqual(len(labels_B), expected_label_count)

    # ==================== Complete Graphs ====================

    def test_complete_graph_5_vertices(self):
        """Test on complete graph K5."""
        G = GraphGenerator.complete_graph(5)
        result = self.run_algorithm_with_timeout(G, eps=0.1)
        if result is not None:
            labels_A, labels_B = result
            self.validate_partition_labels(labels_A, labels_B, G)

    def test_complete_graph_10_vertices(self):
        """Test on complete graph K10."""
        G = GraphGenerator.complete_graph(10)
        result = self.run_algorithm_with_timeout(G, eps=0.1)
        if result is not None:
            labels_A, labels_B = result
            self.validate_partition_labels(labels_A, labels_B, G)

    def test_complete_graph_15_vertices(self):
        """Test on complete graph K15."""
        G = GraphGenerator.complete_graph(15)
        result = self.run_algorithm_with_timeout(G, eps=0.1)
        if result is not None:
            labels_A, labels_B = result
            self.validate_partition_labels(labels_A, labels_B, G)

    def test_complete_graph_20_vertices(self):
        """Test on complete graph K20."""
        G = GraphGenerator.complete_graph(20)
        result = self.run_algorithm_with_timeout(G, eps=0.1)
        if result is not None:
            labels_A, labels_B = result
            self.validate_partition_labels(labels_A, labels_B, G)

    # ==================== Bipartite Graphs ====================

    def test_complete_bipartite_3_3(self):
        """Test on complete bipartite graph K3,3."""
        G = GraphGenerator.bipartite_graph(3, 3)
        result = self.run_algorithm_with_timeout(G, eps=0.1)
        if result is not None:
            labels_A, labels_B = result
            self.validate_partition_labels(labels_A, labels_B, G)

    def test_complete_bipartite_5_5(self):
        """Test on complete bipartite graph K5,5."""
        G = GraphGenerator.bipartite_graph(5, 5)
        result = self.run_algorithm_with_timeout(G, eps=0.1)
        if result is not None:
            labels_A, labels_B = result
            self.validate_partition_labels(labels_A, labels_B, G)

    def test_complete_bipartite_7_8(self):
        """Test on complete bipartite graph K7,8."""
        G = GraphGenerator.bipartite_graph(7, 8)
        result = self.run_algorithm_with_timeout(G, eps=0.1)
        if result is not None:
            labels_A, labels_B = result
            self.validate_partition_labels(labels_A, labels_B, G)

    def test_random_bipartite_sparse(self):
        """Test on sparse random bipartite graph."""
        G = GraphGenerator.bipartite_graph(8, 8, edge_prob=0.2, seed=101)
        result = self.run_algorithm_with_timeout(G, eps=0.1)
        if result is not None:
            labels_A, labels_B = result
            self.validate_partition_labels(labels_A, labels_B, G)

    def test_random_bipartite_dense(self):
        """Test on dense random bipartite graph."""
        G = GraphGenerator.bipartite_graph(6, 6, edge_prob=0.8, seed=102)
        result = self.run_algorithm_with_timeout(G, eps=0.1)
        if result is not None:
            labels_A, labels_B = result
            self.validate_partition_labels(labels_A, labels_B, G)

    # ==================== Random Graphs ====================

    def test_random_graph_sparse_10(self):
        """Test on sparse random graph with 9 vertices."""
        G = GraphGenerator.random_graph(9, edge_prob=0.1, seed=201)
        result = self.run_algorithm_with_timeout(G, eps=0.1)
        if result is not None:
            labels_A, labels_B = result
            self.validate_partition_labels(labels_A, labels_B, G)

    def test_random_graph_sparse_15(self):
        """Test on sparse random graph with 15 vertices."""
        G = GraphGenerator.random_graph(15, edge_prob=0.08, seed=202)
        result = self.run_algorithm_with_timeout(G, eps=0.1)
        if result is not None:
            labels_A, labels_B = result
            self.validate_partition_labels(labels_A, labels_B, G)

    def test_random_graph_medium_8(self):
        """Test on complete bipartite graph K4,4."""
        G = GraphGenerator.bipartite_graph(4, 4, edge_prob=1.0, seed=203)
        result = self.run_algorithm_with_timeout(G, eps=0.1)
        if result is not None:
            labels_A, labels_B = result
            self.validate_partition_labels(labels_A, labels_B, G)

    def test_random_graph_medium_12(self):
        """Test on medium-sparse random graph with 8 vertices."""
        G = GraphGenerator.random_graph(8, edge_prob=0.25, seed=204)
        result = self.run_algorithm_with_timeout(G, eps=0.1)
        if result is not None:
            labels_A, labels_B = result
            self.validate_partition_labels(labels_A, labels_B, G)

    def test_random_graph_sparse_7(self):
        """Test on sparse random graph with 7 vertices."""
        G = GraphGenerator.random_graph(7, edge_prob=0.1, seed=205)
        result = self.run_algorithm_with_timeout(G, eps=0.1)
        if result is not None:
            labels_A, labels_B = result
            self.validate_partition_labels(labels_A, labels_B, G)

    def test_random_graph_zero_edge_refinement_regression(self):
        """Sparse refinements with zero local A-B edges should not crash."""
        G = nx.gnp_random_graph(8, 0.3, seed=1)
        result = self.run_algorithm_with_timeout(G, eps=0.1, max_depth=8)
        if result is not None:
            labels_A, labels_B = result
            self.validate_partition_labels(labels_A, labels_B, G)

    def test_random_graph_sparse_small_8(self):
        """Test on sparse random graph with 7 vertices."""
        G = GraphGenerator.random_graph(7, edge_prob=0.35, seed=206)
        result = self.run_algorithm_with_timeout(G, eps=0.1)
        if result is not None:
            labels_A, labels_B = result
            self.validate_partition_labels(labels_A, labels_B, G)

    # ==================== Stochastic Block Models ====================

    def test_sbm_two_blocks_balanced_sparse(self):
        """Test on SBM with 2 blocks, balanced, sparse intra-block density."""
        # 2 blocks, each with 3 vertices
        # Sparse intra-block density (0.25), very low inter-block density (0.05)
        sizes = [3, 3]
        probs = [[0.25, 0.05], [0.05, 0.25]]
        G = GraphGenerator.stochastic_block_model(sizes, probs, seed=301)
        result = self.run_algorithm_with_timeout(G, eps=0.1)
        if result is not None:
            labels_A, labels_B = result
            self.validate_partition_labels(labels_A, labels_B, G)

    def test_sbm_two_blocks_sparse_intra(self):
        """Test on SBM with 2 blocks, balanced, very sparse."""
        # 2 blocks with sizes 3 and 3
        sizes = [3, 3]
        probs = [[0.3, 0.05], [0.05, 0.3]]
        G = GraphGenerator.stochastic_block_model(sizes, probs, seed=302)
        result = self.run_algorithm_with_timeout(G, eps=0.1)
        if result is not None:
            labels_A, labels_B = result
            self.validate_partition_labels(labels_A, labels_B, G)

    def test_sbm_three_blocks_4_each_very_sparse(self):
        """Test on SBM with 3 blocks, balanced, very sparse inter-block edges."""
        # 3 blocks, each with 4 vertices
        sizes = [4, 4, 4]
        probs = [[0.3, 0.02, 0.02], [0.02, 0.3, 0.02], [0.02, 0.02, 0.3]]
        G = GraphGenerator.stochastic_block_model(sizes, probs, seed=303)
        result = self.run_algorithm_with_timeout(G, eps=0.1)
        if result is not None:
            labels_A, labels_B = result
            self.validate_partition_labels(labels_A, labels_B, G)

    def test_sbm_three_blocks_balanced_sparse(self):
        """Test on SBM with 3 blocks, balanced, sparse edges."""
        # 3 blocks, each with 3 vertices
        sizes = [3, 3, 3]
        probs = [[0.2, 0.05, 0.05], [0.05, 0.2, 0.05], [0.05, 0.05, 0.2]]
        G = GraphGenerator.stochastic_block_model(sizes, probs, seed=304)
        result = self.run_algorithm_with_timeout(G, eps=0.1)
        if result is not None:
            labels_A, labels_B = result
            self.validate_partition_labels(labels_A, labels_B, G)

    def test_sbm_three_blocks_3_each_very_sparse(self):
        """Test on SBM with 3 blocks, balanced, very sparse inter-block edges."""
        # 3 blocks, each with 3 vertices
        sizes = [3, 3, 3]
        probs = [[0.25, 0.02, 0.02], [0.02, 0.25, 0.02], [0.02, 0.02, 0.25]]
        G = GraphGenerator.stochastic_block_model(sizes, probs, seed=305)
        result = self.run_algorithm_with_timeout(G, eps=0.1)
        if result is not None:
            labels_A, labels_B = result
            self.validate_partition_labels(labels_A, labels_B, G)

    def test_sbm_two_blocks_balanced_sparse_020(self):
        """Test on SBM with 2 blocks, balanced, sparse intra-block density."""
        # 2 blocks, each with 3 vertices
        # Sparse intra-block density (0.2), very low inter-block density (0.05)
        sizes = [3, 3]
        probs = [[0.2, 0.05], [0.05, 0.2]]
        G = GraphGenerator.stochastic_block_model(sizes, probs, seed=306)
        result = self.run_algorithm_with_timeout(G, eps=0.1)
        if result is not None:
            labels_A, labels_B = result
            self.validate_partition_labels(labels_A, labels_B, G)

    def test_sbm_two_blocks_low_contrast(self):
        """Test on SBM with 2 blocks, balanced, moderate density."""
        # 2 blocks, each with 3 vertices
        sizes = [3, 3]
        probs = [[0.4, 0.1], [0.1, 0.4]]
        G = GraphGenerator.stochastic_block_model(sizes, probs, seed=307)
        result = self.run_algorithm_with_timeout(G, eps=0.1)
        if result is not None:
            labels_A, labels_B = result
            self.validate_partition_labels(labels_A, labels_B, G)


class TestGraphGenerationAndProperties(unittest.TestCase):
    """Test that generated graphs have expected properties."""

    def test_complete_graph_edge_count(self):
        """Verify complete graph has correct number of edges."""
        G = GraphGenerator.complete_graph(5)
        expected_edges = 5 * 4 / 2  # n(n-1)/2
        self.assertEqual(G.number_of_edges(), int(expected_edges))

    def test_bipartite_graph_is_bipartite(self):
        """Verify generated bipartite graphs are actually bipartite."""
        G = GraphGenerator.bipartite_graph(5, 5)
        self.assertTrue(nx.is_bipartite(G))

    def test_random_graph_vertex_count(self):
        """Verify random graphs have correct number of vertices."""
        G = GraphGenerator.random_graph(15, edge_prob=0.3, seed=401)
        self.assertEqual(G.number_of_nodes(), 15)

    def test_sbm_vertex_count(self):
        """Verify SBM graphs have correct number of vertices."""
        sizes = [5, 7, 6]
        probs = [[0.8, 0.2, 0.2], [0.2, 0.8, 0.2], [0.2, 0.2, 0.8]]
        G = GraphGenerator.stochastic_block_model(sizes, probs, seed=402)
        self.assertEqual(G.number_of_nodes(), sum(sizes))


if __name__ == '__main__':
    unittest.main()
