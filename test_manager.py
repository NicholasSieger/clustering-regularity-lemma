#!/usr/bin/env python3
"""Unit tests for manager.py - FileManager and GraphManager classes"""

import importlib.util
import json
import os
import tempfile
import shutil
import numpy as np
import networkx as nx
import unittest

# Load manager.py module
spec = importlib.util.spec_from_file_location("manager", "manager.py")
manager_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manager_module)

FileManager = manager_module.FileManager
GraphManager = manager_module.GraphManager


class TestFileManager(unittest.TestCase):
    """Test FileManager class."""
    
    def setUp(self):
        """Create temporary directories for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.partition_dir = os.path.join(self.temp_dir, "partitions")
        self.graph_dir = os.path.join(self.temp_dir, "graphs")
        self.file_manager = FileManager(self.partition_dir, self.graph_dir)
    
    def tearDown(self):
        """Clean up temporary directories."""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test FileManager initialization creates directories."""
        self.assertTrue(os.path.exists(self.partition_dir))
        self.assertTrue(os.path.exists(self.graph_dir))
        self.assertEqual(self.file_manager.target_dir, self.partition_dir)
        self.assertEqual(self.file_manager.graph_dir, self.graph_dir)
    
    def test_partition_file_name(self):
        """Test partitionFileName generates correct path."""
        filename = self.file_manager.partitionFileName(42, "down")
        basename = os.path.basename(filename)

        self.assertEqual(os.path.dirname(filename), self.partition_dir)
        self.assertTrue(basename.startswith("partition_42_"))
        self.assertTrue(basename.endswith(".json"))
        self.assertNotIn("down", basename)
        self.assertEqual(len(basename), len("partition_42_") + 16 + len(".json"))
    
    def test_save_and_load_partition(self):
        """Test savePartition and loadPartition work correctly."""
        vertex = 5
        direction = "up"
        A = np.array([1, 2, 3])
        B = np.array([4, 5, 6])
        mask_A = np.array([True, False, True])
        mask_B = np.array([False, True, True])
        
        # Save partition
        self.file_manager.savePartition(vertex, direction, mask_A, mask_B, A, B)
        
        # Load partition
        success, data = self.file_manager.loadPartition(vertex, direction)
        
        self.assertTrue(success)
        self.assertIsNotNone(data)
        partition_dict = json.loads(data)
        self.assertEqual(set(partition_dict.keys()), {
            "vtx",
            "dir",
            "mask_A",
            "mask_B",
            "neighbors_A",
            "neighbors_B",
        })
        self.assertNotIn("A", partition_dict)
        self.assertNotIn("B", partition_dict)
        self.assertEqual(partition_dict["vtx"], vertex)
        self.assertEqual(partition_dict["dir"], direction)
        self.assertTrue(np.array_equal(np.array(partition_dict["mask_A"], dtype=bool), mask_A))
        self.assertTrue(np.array_equal(np.array(partition_dict["mask_B"], dtype=bool), mask_B))
        self.assertTrue(np.array_equal(np.array(partition_dict["neighbors_A"]), A))
        self.assertTrue(np.array_equal(np.array(partition_dict["neighbors_B"]), B))
    
    def test_load_partition_not_exists(self):
        """Test loadPartition returns False for non-existent file."""
        success, data = self.file_manager.loadPartition(999, "nonexistent")
        self.assertFalse(success)
        self.assertIsNone(data)
    
    def test_graph_file_name(self):
        """Test graphFileName generates correct path."""
        filename = self.file_manager.graphFileName(10)
        expected = os.path.join(self.graph_dir, "linkGraph_10.json")
        self.assertEqual(filename, expected)
    
    def test_save_and_load_link_graph(self):
        """Test saveLinkGraph and loadLinkGraph work correctly."""
        vertex = 7
        G = nx.Graph()
        G.add_nodes_from([1, 2, 3])
        G.add_edges_from([(1, 2), (2, 3)])
        
        # Save graph
        self.file_manager.saveLinkGraph(vertex, G)
        
        # Load graph
        success, loaded_G = self.file_manager.loadLinkGraph(vertex)
        
        self.assertTrue(success)
        self.assertIsInstance(loaded_G, nx.Graph)
        self.assertEqual(set(loaded_G.nodes()), set(G.nodes()))
        self.assertEqual(set(loaded_G.edges()), set(G.edges()))
    
    def test_load_link_graph_not_exists(self):
        """Test loadLinkGraph returns False for non-existent file."""
        success, data = self.file_manager.loadLinkGraph(999)
        self.assertFalse(success)
        self.assertIsNone(data)


class TestGraphManager(unittest.TestCase):
    """Test GraphManager class."""
    
    def setUp(self):
        """Create a simple graph for testing."""
        self.G = nx.Graph()
        self.G.add_nodes_from([0, 1, 2, 3])
        self.G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
        self.graph_manager = GraphManager(self.G)
    
    def test_initialization(self):
        """Test GraphManager initialization creates tripartite graph."""
        H = self.graph_manager.H
        self.assertIsNotNone(H)
        # Should have 3 copies of each original node
        self.assertEqual(len(H.nodes()), len(self.G.nodes()) * 3)
    
    def test_make_tripartite(self):
        """Test makeTripartite creates correct tripartite structure."""
        H = self.graph_manager.makeTripartite(self.G)
        
        # Check all nodes have part attribute
        for node, data in H.nodes(data=True):
            self.assertIn("part", data)
            self.assertIn(data["part"], [0, 1, 2])
        
        # Check edges exist between all parts
        expected_edges = 0
        for i, j in [(0, 1), (1, 2), (2, 0)]:
            expected_edges += len(self.G.edges())
        self.assertEqual(len(H.edges()), expected_edges * 2)
    
    def test_get_v2(self):
        """Test getV2 returns correct vertices."""
        V2 = self.graph_manager.getV2()
        
        # Should return nodes with part=1 (as integers)
        self.assertEqual(len(V2), len(self.G.nodes()))
        for node in V2:
            self.assertIsInstance(node, int)
    
    def test_make_link_graph(self):
        """Test makeLinkGraph creates correct link graph."""
        vertex = 1
        link_graph = self.graph_manager.makeLinkGraph(vertex)
        
        self.assertIsInstance(link_graph, nx.Graph)
        # Link graph should contain neighbors of (vertex, 1)
        self.assertGreater(len(link_graph.nodes()), 0)
    
    def test_make_link_partition(self):
        """Test makeLinkPartition returns correct partitions."""
        vertex = 1
        A, B = self.graph_manager.makeLinkPartition(vertex)
        
        self.assertIsInstance(A, list)
        self.assertIsInstance(B, list)
        
        # A should contain part=0 nodes, B should contain part=2 nodes
        for node in A:
            self.assertEqual(node[1], 0)
        for node in B:
            self.assertEqual(node[1], 2)
    
    def test_empty_graph(self):
        """Test GraphManager with empty graph."""
        empty_G = nx.Graph()
        gm = GraphManager(empty_G)
        
        V2 = gm.getV2()
        self.assertEqual(len(V2), 0)
    
    def test_single_node_graph(self):
        """Test GraphManager with single node graph."""
        single_G = nx.Graph()
        single_G.add_node(0)
        gm = GraphManager(single_G)
        
        self.assertEqual(len(gm.H.nodes()), 3)
        V2 = gm.getV2()
        self.assertEqual(len(V2), 1)


class TestFileManagerDeletePartition(unittest.TestCase):
    """Test FileManager.deletePartition method."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.partition_dir = os.path.join(self.temp_dir, "partitions")
        self.graph_dir = os.path.join(self.temp_dir, "graphs")
        self.file_manager = FileManager(self.partition_dir, self.graph_dir)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_delete_partition_existing(self):
        """Test deleting an existing partition file."""
        vertex = 10
        direction = "test"
        A = np.array([1, 2, 3])
        B = np.array([4, 5, 6])
        mask_A = np.ones(len(A), dtype=bool)
        mask_B = np.ones(len(B), dtype=bool)
        
        # Create and verify partition exists
        self.file_manager.savePartition(vertex, direction, mask_A, mask_B, A, B)
        self.assertTrue(os.path.exists(self.file_manager.partitionFileName(vertex, direction)))
        
        # Delete and verify
        self.file_manager.deletePartition(vertex, direction)
        self.assertFalse(os.path.exists(self.file_manager.partitionFileName(vertex, direction)))
    
    def test_delete_partition_not_exists(self):
        """Test deleting a non-existent partition does not raise."""
        # Should not raise an exception
        self.file_manager.deletePartition(999, "nonexistent")


class TestManager(unittest.TestCase):
    """Test Manager class."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.partition_dir = os.path.join(self.temp_dir, "partitions")
        self.graph_dir = os.path.join(self.temp_dir, "graphs")
        
        # Create a simple graph
        self.G = nx.Graph()
        self.G.add_nodes_from([0, 1, 2, 3])
        self.G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
        
        self.eps = 0.1
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def create_manager(self):
        Manager = manager_module.Manager
        return Manager(
            self.G,
            self.eps,
            partition_dir=self.partition_dir,
            graph_dir=self.graph_dir,
        )
    
    def test_manager_initialization(self):
        """Test Manager initialization."""
        mgr = self.create_manager()

        self.assertEqual(mgr.eps, self.eps)
        self.assertIsInstance(mgr.graph_manager, GraphManager)
        self.assertIsInstance(mgr.partition_manager, FileManager)
        self.assertIsNotNone(mgr.q)
        self.assertGreater(len(mgr.V2), 0)
    
    def test_manager_parameters(self):
        """Test Manager sets correct parameters."""
        mgr = self.create_manager()

        self.assertEqual(mgr.eps, self.eps)
        self.assertEqual(mgr.irreg_vtx_threshold, self.eps**5 / 90)
        self.assertEqual(mgr.dev_vtx_threshold, self.eps**2 / 9)
        self.assertEqual(mgr.dev_split_threshold, self.eps**5)
        self.assertEqual(mgr.irreg_vtx_count_threshold, self.eps**(5/2) / 9)
        self.assertEqual(mgr.dev_threshold, 2 * self.eps**2 / 5)
        self.assertEqual(mgr.clustering_threshold, self.eps)
    
    def test_manager_saves_initial_partitions(self):
        """Test Manager saves initial partitions for all V2 vertices."""
        mgr = self.create_manager()
        mgr._initialize_link_data()

        for v in mgr.V2:
            success, partition = mgr.partition_manager.loadPartition(v, "")
            self.assertTrue(success, f"Partition not saved for vertex {v}")

            success, link = mgr.partition_manager.loadLinkGraph(v)
            self.assertTrue(success, f"Link graph not saved for vertex {v}")
    
    def test_compute_path_data(self):
        """Test compute_path_data method."""
        mgr = self.create_manager()
        mgr._initialize_link_data()

        pathweight, triangle_count, gamma = mgr.compute_path_data("")

        self.assertIsInstance(pathweight, (int, float))
        self.assertIsInstance(triangle_count, int)
        self.assertIsInstance(gamma, (int, float))
        self.assertGreaterEqual(triangle_count, 0)
    
    def test_compute_path_data_empty_dir(self):
        """Test compute_path_data with non-existent directory."""
        mgr = self.create_manager()

        pathweight, triangle_count, gamma = mgr.compute_path_data("nonexistent")
        self.assertEqual(pathweight, 0)
        self.assertEqual(triangle_count, 0)
        self.assertEqual(gamma, 0.0)

    def test_iterate_quietly_handles_unsaved_direction(self):
        """Queued directions with no saved partitions are zero-work branches."""
        mgr = self.create_manager()
        mgr.q = __import__("queue").Queue()
        mgr.q.put("unsaved")

        out = mgr.iterate()

        self.assertEqual(out, ["unsaved"])
        self.assertEqual(mgr.directions_considered, ["unsaved"])
        self.assertEqual(mgr.partition_logs[-1]["pathweight"], 0)
        self.assertEqual(mgr.partition_logs[-1]["triangle_count"], 0)
        self.assertEqual(mgr.partition_logs[-1]["gamma"], 0.0)
        self.assertEqual(mgr.partition_logs[-1]["failure_reason"], "PASSED_GAMMA_CHECK")

    def test_iterate_method_exists(self):
        """Test iterate method exists."""
        mgr = self.create_manager()

        self.assertTrue(hasattr(mgr, 'iterate'))
        self.assertTrue(callable(mgr.iterate))


if __name__ == '__main__':
    unittest.main()
