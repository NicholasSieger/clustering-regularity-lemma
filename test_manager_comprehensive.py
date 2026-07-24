#!/usr/bin/env python3
"""
Comprehensive unit tests for manager.py

Tests focus on:
- FileIO errors and robustness
- Data leaking (files created but not deleted)
- Index errors
- Max_depth parameter ensures full cleanup
- Bitmask disjointness in assemble_partition
"""

import pytest
import os
import shutil
import tempfile
import json
import networkx as nx
import numpy as np
from pathlib import Path

from algorithm import AlgorithmRunner
from manager import AlgorithmParameters, EdgePartitionAssembler, Manager, FileManager, GraphManager, PartitionRecord


@pytest.fixture
def temp_dirs():
    """Create temporary directories for partitions and graphs."""
    partition_dir = tempfile.mkdtemp(prefix="test_partitions_")
    graph_dir = tempfile.mkdtemp(prefix="test_graphs_")
    yield partition_dir, graph_dir
    # Cleanup after test
    shutil.rmtree(partition_dir, ignore_errors=True)
    shutil.rmtree(graph_dir, ignore_errors=True)


@pytest.fixture
def small_complete_graph():
    """Create a small complete graph K4."""
    return nx.complete_graph(4)


@pytest.fixture
def medium_complete_graph():
    """Create a medium complete graph K6."""
    return nx.complete_graph(6)


def save_full_partition(file_manager, vertex, direction, neighbors_A, neighbors_B):
    """Save a partition that marks every provided neighbor."""
    mask_A = np.ones(len(neighbors_A), dtype=bool)
    mask_B = np.ones(len(neighbors_B), dtype=bool)
    file_manager.savePartition(vertex, direction, mask_A, mask_B, neighbors_A, neighbors_B)


class TestRefactoredDataTypes:
    """Tests for explicit refactor data types."""

    def test_algorithm_parameters_defaults(self):
        params = AlgorithmParameters(eps=0.25)

        assert params.irreg_vtx_threshold == 0.25**5 / 90
        assert params.dev_vtx_threshold == 0.25**2 / 9
        assert params.irreg_vtx_count_threshold == 0.25**(5/2) / 9
        assert params.dev_threshold == 2 * 0.25**2 / 5
        assert params.dev_split_threshold == 0.25**5
        assert params.irreg_threshold == 2 * 0.25**(5/2) / 5
        assert params.clustering_threshold == 0.25

    def test_partition_record_json_round_trip(self):
        record = PartitionRecord(
            vertex=7,
            direction="i0",
            mask_A=np.array([True, False, True]),
            mask_B=np.array([False, True]),
            neighbors_A=[(1, 0), (2, 0), (3, 0)],
            neighbors_B=[(4, 2), (5, 2)],
        )

        data = record.to_json_dict()
        assert set(data) == {"vtx", "dir", "mask_A", "mask_B", "neighbors_A", "neighbors_B"}
        assert "A" not in data
        assert "B" not in data

        loaded = PartitionRecord.from_json_dict(data)
        assert loaded.vertex == record.vertex
        assert loaded.direction == record.direction
        assert np.array_equal(loaded.mask_A, record.mask_A)
        assert np.array_equal(loaded.mask_B, record.mask_B)
        assert loaded.neighbors_A == record.neighbors_A
        assert loaded.neighbors_B == record.neighbors_B

    def test_partition_record_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="mask_A length"):
            PartitionRecord(
                vertex=0,
                direction="bad",
                mask_A=np.array([True, False]),
                mask_B=np.array([True]),
                neighbors_A=[(1, 0)],
                neighbors_B=[(1, 2)],
            )


class TestFileManager:
    """Tests for FileManager class."""

    def test_filemanager_init_creates_directories(self, temp_dirs):
        """FileManager should create directories if they don't exist."""
        partition_dir, graph_dir = temp_dirs
        # Clean up one directory to test creation
        shutil.rmtree(partition_dir)
        assert not os.path.exists(partition_dir)
        
        fm = FileManager(partition_dir, graph_dir)
        assert os.path.exists(partition_dir)
        assert os.path.exists(graph_dir)

    def test_partition_filename_generation(self, temp_dirs):
        """Test that partition filenames are generated correctly."""
        partition_dir, graph_dir = temp_dirs
        fm = FileManager(partition_dir, graph_dir)
        
        filename = fm.partitionFileName(5, "i0")
        basename = Path(filename).name
        assert Path(filename).parent == Path(partition_dir)
        assert basename.startswith("partition_5_")
        assert basename.endswith(".json")
        assert "i0" not in basename
        assert len(basename) == len("partition_5_") + 16 + len(".json")

    def test_long_direction_filename_is_bounded(self, temp_dirs):
        """Long direction codes should not be embedded in partition filenames."""
        partition_dir, graph_dir = temp_dirs
        fm = FileManager(partition_dir, graph_dir)
        direction = "i0" * 200

        filename = fm.partitionFileName(12, direction)
        basename = Path(filename).name

        assert Path(filename).parent == Path(partition_dir)
        assert basename.startswith("partition_12_")
        assert basename.endswith(".json")
        assert direction not in basename
        assert len(basename) == len("partition_12_") + 16 + len(".json")

    def test_long_direction_round_trip_preserves_json_direction(self, temp_dirs):
        """Hashed filenames should not truncate the persisted direction value."""
        partition_dir, graph_dir = temp_dirs
        fm = FileManager(partition_dir, graph_dir)
        direction = "d3i1" * 100
        neighbors_A = np.array([0, 1])
        neighbors_B = np.array([2, 3])
        mask_A = np.array([True, False])
        mask_B = np.array([False, True])

        fm.savePartition(7, direction, mask_A, mask_B, neighbors_A, neighbors_B)
        success, partition_str = fm.loadPartition(7, direction)

        assert success is True
        partition_dict = json.loads(partition_str)
        assert partition_dict["dir"] == direction
        assert np.array_equal(np.array(partition_dict["mask_A"], dtype=bool), mask_A)
        assert np.array_equal(np.array(partition_dict["mask_B"], dtype=bool), mask_B)

    def test_different_directions_get_different_filenames(self, temp_dirs):
        """Direction hashes should distinguish partitions for the same vertex."""
        partition_dir, graph_dir = temp_dirs
        fm = FileManager(partition_dir, graph_dir)

        first = fm.partitionFileName(3, "i0")
        second = fm.partitionFileName(3, "i1")

        assert first != second

    def test_delete_partition_with_long_direction(self, temp_dirs):
        """deletePartition should use the same direction hash as savePartition."""
        partition_dir, graph_dir = temp_dirs
        fm = FileManager(partition_dir, graph_dir)
        direction = "i0d1" * 100
        A = np.array([0, 1])
        B = np.array([2, 3])

        save_full_partition(fm, 4, direction, A, B)
        filename = fm.partitionFileName(4, direction)
        assert os.path.exists(filename)

        fm.deletePartition(4, direction)
        assert not os.path.exists(filename)

    def test_save_and_load_partition(self, temp_dirs):
        """Test saving and loading partitions."""
        partition_dir, graph_dir = temp_dirs
        fm = FileManager(partition_dir, graph_dir)
        
        neighbors_A = np.array([0, 1, 2])
        neighbors_B = np.array([3, 4, 5])
        mask_A = np.array([True, False, True])
        mask_B = np.array([False, True, True])
        
        fm.savePartition(0, "test", mask_A, mask_B, neighbors_A, neighbors_B)
        success, partition_str = fm.loadPartition(0, "test")
        
        assert success is True
        assert partition_str is not None
        
        partition_dict = json.loads(partition_str)
        assert set(partition_dict.keys()) == {
            "vtx",
            "dir",
            "mask_A",
            "mask_B",
            "neighbors_A",
            "neighbors_B",
        }
        assert "A" not in partition_dict
        assert "B" not in partition_dict
        assert np.array_equal(np.array(partition_dict["mask_A"], dtype=bool), mask_A)
        assert np.array_equal(np.array(partition_dict["mask_B"], dtype=bool), mask_B)
        assert np.array_equal(np.array(partition_dict["neighbors_A"]), neighbors_A)
        assert np.array_equal(np.array(partition_dict["neighbors_B"]), neighbors_B)

    def test_load_nonexistent_partition(self, temp_dirs):
        """Test loading a partition that doesn't exist."""
        partition_dir, graph_dir = temp_dirs
        fm = FileManager(partition_dir, graph_dir)
        
        success, partition_str = fm.loadPartition(999, "nonexistent")
        assert success is False
        assert partition_str is None

    def test_delete_partition(self, temp_dirs):
        """Test deleting a partition."""
        partition_dir, graph_dir = temp_dirs
        fm = FileManager(partition_dir, graph_dir)
        
        A = np.array([0, 1])
        B = np.array([2, 3])
        save_full_partition(fm, 0, "test", A, B)
        
        filename = fm.partitionFileName(0, "test")
        assert os.path.exists(filename)
        
        fm.deletePartition(0, "test")
        assert not os.path.exists(filename)

    def test_delete_nonexistent_partition(self, temp_dirs):
        """Deleting a nonexistent partition should not raise an error."""
        partition_dir, graph_dir = temp_dirs
        fm = FileManager(partition_dir, graph_dir)
        
        # Should not raise
        fm.deletePartition(999, "nonexistent")

    def test_deleteAllPartitions_removes_all_files(self, temp_dirs):
        """deleteAllPartitions should remove all partition files."""
        partition_dir, graph_dir = temp_dirs
        fm = FileManager(partition_dir, graph_dir)
        
        A = np.array([0, 1])
        B = np.array([2, 3])
        
        # Create multiple partitions
        for i in range(3):
            save_full_partition(fm, i, f"dir{i}", A, B)
        
        assert len(os.listdir(partition_dir)) == 3
        fm.deleteAllPartitions()
        assert len(os.listdir(partition_dir)) == 0

    def test_save_and_load_link_graph(self, temp_dirs):
        """Test saving and loading link graphs."""
        partition_dir, graph_dir = temp_dirs
        fm = FileManager(partition_dir, graph_dir)
        
        G = nx.complete_graph(4)
        fm.saveLinkGraph(0, G)
        
        success, loaded_G = fm.loadLinkGraph(0)
        assert success is True
        assert nx.is_isomorphic(G, loaded_G)

    def test_load_nonexistent_link_graph(self, temp_dirs):
        """Test loading a link graph that doesn't exist."""
        partition_dir, graph_dir = temp_dirs
        fm = FileManager(partition_dir, graph_dir)
        
        success, loaded_G = fm.loadLinkGraph(999)
        assert success is False
        assert loaded_G is None

    def test_deleteAllLinkGraphs_removes_all_files(self, temp_dirs):
        """deleteAllLinkGraphs should remove all graph files."""
        partition_dir, graph_dir = temp_dirs
        fm = FileManager(partition_dir, graph_dir)
        
        G = nx.complete_graph(4)
        for i in range(3):
            fm.saveLinkGraph(i, G)
        
        assert len(os.listdir(graph_dir)) == 3
        fm.deleteAllLinkGraphs()
        assert len(os.listdir(graph_dir)) == 0


class TestGraphManager:
    """Tests for GraphManager class."""

    def test_graph_manager_creates_tripartite(self, small_complete_graph):
        """GraphManager should create a tripartite cover."""
        gm = GraphManager(small_complete_graph)
        
        # Check that tripartite graph has 3 parts
        parts = {}
        for node, data in gm.H.nodes(data=True):
            part = data.get("part")
            if part not in parts:
                parts[part] = []
            parts[part].append(node)
        
        assert len(parts) == 3
        assert 0 in parts and 1 in parts and 2 in parts

    def test_getV2_returns_correct_vertices(self, small_complete_graph):
        """getV2 should return vertices with part=1."""
        gm = GraphManager(small_complete_graph)
        V2 = gm.getV2()
        
        assert len(V2) == len(small_complete_graph)
        assert all(isinstance(v, int) for v in V2)

    def test_makeLinkGraph(self, small_complete_graph):
        """makeLinkGraph should return valid subgraph."""
        gm = GraphManager(small_complete_graph)
        V2 = gm.getV2()
        
        link_graph = gm.makeLinkGraph(V2[0])
        assert isinstance(link_graph, nx.Graph)
        assert link_graph.number_of_nodes() > 0

    def test_makeLinkPartition(self, small_complete_graph):
        """makeLinkPartition should return two non-empty neighbor lists."""
        gm = GraphManager(small_complete_graph)
        V2 = gm.getV2()
        
        A, B = gm.makeLinkPartition(V2[0])
        assert isinstance(A, list)
        assert isinstance(B, list)
        assert len(A) > 0 and len(B) > 0


class TestManagerInitialization:
    """Tests for Manager class initialization."""

    def test_manager_initialization(self, small_complete_graph, temp_dirs):
        """Manager should initialize with all hyperparameters."""
        partition_dir, graph_dir = temp_dirs
        eps = 0.5
        
        manager = Manager(
            G=small_complete_graph,
            eps=eps,
            irreg_vtx_threshold=eps**5 / 90,
            dev_vtx_threshold=eps,
            irreg_vtx_count_threshold=0.1,
            dev_threshold=0.1,
            irreg_threshold=eps,
            clustering_threshold=eps,
            max_depth=5
        )
        
        assert manager.eps == eps
        assert manager.max_depth == 5
        assert isinstance(manager.graph_manager, GraphManager)
        assert isinstance(manager.partition_manager, FileManager)

    def test_manager_initialization_with_infinite_max_depth(self, small_complete_graph, temp_dirs):
        """Manager should accept infinite max_depth."""
        partition_dir, graph_dir = temp_dirs
        
        manager = Manager(
            G=small_complete_graph,
            eps=0.5,
            irreg_vtx_threshold=0.5**5 / 90,
            dev_vtx_threshold=0.5,
            irreg_vtx_count_threshold=0.1,
            dev_threshold=0.1,
            irreg_threshold=0.5,
            clustering_threshold=0.5,
            max_depth=float('inf')
        )
        
        assert manager.max_depth == float('inf')

    def test_algorithm_runner_overrides_max_depth_when_parameters_are_supplied(self, small_complete_graph, temp_dirs):
        """Explicit max_depth should override the value on a supplied parameter object."""
        partition_dir, graph_dir = temp_dirs
        params = AlgorithmParameters(eps=0.25)

        runner = AlgorithmRunner(
            small_complete_graph,
            parameters=params,
            max_depth=7,
            partition_dir=partition_dir,
            graph_dir=graph_dir,
        )

        assert runner.max_depth == 7
        assert runner.parameters.max_depth == 7

    def test_algorithm_runner_overrides_thresholds_when_parameters_are_supplied(self, small_complete_graph, temp_dirs):
        """Explicit threshold arguments should override supplied parameter values."""
        partition_dir, graph_dir = temp_dirs
        params = AlgorithmParameters(eps=0.25)

        runner = AlgorithmRunner(
            small_complete_graph,
            parameters=params,
            irreg_vtx_threshold=0.11,
            dev_threshold=0.22,
            dev_split_threshold=0.33,
            partition_dir=partition_dir,
            graph_dir=graph_dir,
        )

        assert runner.irreg_vtx_threshold == 0.11
        assert runner.dev_threshold == 0.22
        assert runner.dev_split_threshold == 0.33

    def test_algorithm_runner_rejects_conflicting_eps_when_parameters_are_supplied(self, small_complete_graph, temp_dirs):
        """The runner should not silently merge parameter objects with a different eps."""
        partition_dir, graph_dir = temp_dirs
        params = AlgorithmParameters(eps=0.25)

        with pytest.raises(ValueError, match="conflicting eps"):
            AlgorithmRunner(
                small_complete_graph,
                eps=0.5,
                parameters=params,
                partition_dir=partition_dir,
                graph_dir=graph_dir,
            )

    def test_manager_stores_hyperparameters(self, small_complete_graph, temp_dirs):
        """Manager should store all hyperparameters."""
        partition_dir, graph_dir = temp_dirs
        eps = 0.25
        
        irreg_vtx_thresh = eps**5 / 90
        dev_vtx_thresh = eps
        irreg_count_thresh = 0.1
        dev_thresh = 0.1
        dev_split_thresh = eps**5
        irreg_thresh = eps
        clust_thresh = eps
        
        manager = Manager(
            G=small_complete_graph,
            eps=eps,
            irreg_vtx_threshold=irreg_vtx_thresh,
            dev_vtx_threshold=dev_vtx_thresh,
            irreg_vtx_count_threshold=irreg_count_thresh,
            dev_threshold=dev_thresh,
            dev_split_threshold=dev_split_thresh,
            irreg_threshold=irreg_thresh,
            clustering_threshold=clust_thresh,
            max_depth=10
        )
        
        assert manager.eps == eps
        assert manager.irreg_vtx_threshold == irreg_vtx_thresh
        assert manager.dev_vtx_threshold == dev_vtx_thresh
        assert manager.irreg_vtx_count_threshold == irreg_count_thresh
        assert manager.dev_threshold == dev_thresh
        assert manager.dev_split_threshold == dev_split_thresh
        assert manager.irreg_threshold == irreg_thresh
        assert manager.clustering_threshold == clust_thresh


class TestComputeDirectionCodeLength:
    """Tests for compute_direction_code_length helper function."""

    def test_empty_direction_code(self, small_complete_graph):
        """Empty direction code should have length 0."""
        manager = Manager(
            G=small_complete_graph,
            eps=0.5,
            irreg_vtx_threshold=0.5**5 / 90,
            dev_vtx_threshold=0.5,
            irreg_vtx_count_threshold=0.1,
            dev_threshold=0.1,
            irreg_threshold=0.5,
            clustering_threshold=0.5
        )
        
        assert manager.compute_direction_code_length("") == 0

    def test_single_character_direction_code(self, small_complete_graph):
        """Single character direction code."""
        manager = Manager(
            G=small_complete_graph,
            eps=0.5,
            irreg_vtx_threshold=0.5**5 / 90,
            dev_vtx_threshold=0.5,
            irreg_vtx_count_threshold=0.1,
            dev_threshold=0.1,
            irreg_threshold=0.5,
            clustering_threshold=0.5
        )
        
        assert manager.compute_direction_code_length("i") == 1
        assert manager.compute_direction_code_length("d") == 1

    def test_complex_direction_code(self, small_complete_graph):
        """Complex direction code with multiple characters."""
        manager = Manager(
            G=small_complete_graph,
            eps=0.5,
            irreg_vtx_threshold=0.5**5 / 90,
            dev_vtx_threshold=0.5,
            irreg_vtx_count_threshold=0.1,
            dev_threshold=0.1,
            irreg_threshold=0.5,
            clustering_threshold=0.5
        )
        
        assert manager.compute_direction_code_length("i0d1i2") == 6


class TestFileIOCleanup:
    """Tests for ensuring proper file cleanup and avoiding data leaks."""

    def test_no_files_leaked_after_initialization(self, temp_dirs):
        """After initialization, only initial partition and graph files should exist."""
        partition_dir, graph_dir = temp_dirs
        
        G = nx.complete_graph(3)
        manager = Manager(
            G=G,
            eps=0.5,
            irreg_vtx_threshold=0.5**5 / 90,
            dev_vtx_threshold=0.5,
            irreg_vtx_count_threshold=0.1,
            dev_threshold=0.1,
            irreg_threshold=0.5,
            clustering_threshold=0.5,
            partition_dir=partition_dir,
            graph_dir=graph_dir,
        )
        
        manager.V2 = manager.graph_manager.getV2()
        partition_files_before = set(os.listdir(manager.partition_manager.target_dir))
        graph_files_before = set(os.listdir(manager.partition_manager.graph_dir))
        
        for v in manager.V2:
            A, B = manager.graph_manager.makeLinkPartition(v)
            save_full_partition(manager.partition_manager, v, "", A, B)
            N = manager.graph_manager.makeLinkGraph(v)
            manager.partition_manager.saveLinkGraph(v, N)
        
        # Check files were created
        partition_files_after = set(os.listdir(manager.partition_manager.target_dir))
        graph_files_after = set(os.listdir(manager.partition_manager.graph_dir))
        
        assert len(partition_files_after) > 0
        assert len(graph_files_after) > 0
        
        # Cleanup
        manager.partition_manager.deleteAllPartitions()
        manager.partition_manager.deleteAllLinkGraphs()
        
        assert len(os.listdir(manager.partition_manager.target_dir)) == 0
        assert len(os.listdir(manager.partition_manager.graph_dir)) == 0

    def test_max_depth_ensures_full_cleanup(self, small_complete_graph, temp_dirs):
        """When max_depth is exceeded, all files should be cleaned up."""
        partition_dir, graph_dir = temp_dirs
        
        try:
            manager = Manager(
                G=small_complete_graph,
                eps=0.5,
                irreg_vtx_threshold=0.5**5 / 90,
                dev_vtx_threshold=0.5,
                irreg_vtx_count_threshold=0.1,
                dev_threshold=0.1,
                irreg_threshold=0.5,
                clustering_threshold=0.5,
                partition_dir=partition_dir,
                graph_dir=graph_dir,
                max_depth=0  # Set max_depth to 0 so even empty direction "" will fail
            )
            
            # Manually set up some files
            manager.V2 = manager.graph_manager.getV2()
            for v in manager.V2:
                A, B = manager.graph_manager.makeLinkPartition(v)
                save_full_partition(manager.partition_manager, v, "", A, B)
                N = manager.graph_manager.makeLinkGraph(v)
                manager.partition_manager.saveLinkGraph(v, N)
            
            # Verify files exist
            assert len(os.listdir(partition_dir)) > 0
            assert len(os.listdir(graph_dir)) > 0
            
            # Initialize queue with empty direction which will immediately exceed depth
            manager.q = __import__('queue').Queue()
            manager.q.put("i")  # This has length 1, which exceeds max_depth of 0
            
            with pytest.raises(ValueError, match="exceeds maximum depth"):
                manager.iterate()
            
            # Verify cleanup happened - should be empty after error
            assert len(os.listdir(partition_dir)) == 0
            assert len(os.listdir(graph_dir)) == 0
        
        finally:
            # Additional cleanup just in case
            if os.path.exists(partition_dir):
                shutil.rmtree(partition_dir, ignore_errors=True)
            if os.path.exists(graph_dir):
                shutil.rmtree(graph_dir, ignore_errors=True)


class TestPartitionLabels:
    """Tests for partition label generation."""

    def test_partition_labels_disjoint_bitmasks(self, small_complete_graph):
        """partitionLabels should correctly assign labels from disjoint bitmasks."""
        manager = Manager(
            G=small_complete_graph,
            eps=0.5,
            irreg_vtx_threshold=0.5**5 / 90,
            dev_vtx_threshold=0.5,
            irreg_vtx_count_threshold=0.1,
            dev_threshold=0.1,
            irreg_threshold=0.5,
            clustering_threshold=0.5
        )
        
        # Create disjoint bitmasks
        bitmask0 = np.array([True, False, False, False, True])
        bitmask1 = np.array([False, True, False, True, False])
        bitmask2 = np.array([False, False, True, False, False])
        
        bitmasks = [bitmask0, bitmask1, bitmask2]
        labels = manager.partitionLabels(bitmasks)
        
        expected = np.array([0, 1, 2, 1, 0])
        assert np.array_equal(labels, expected)

    def test_partition_labels_all_in_one_class(self, small_complete_graph):
        """When all indices are in one bitmask."""
        manager = Manager(
            G=small_complete_graph,
            eps=0.5,
            irreg_vtx_threshold=0.5**5 / 90,
            dev_vtx_threshold=0.5,
            irreg_vtx_count_threshold=0.1,
            dev_threshold=0.1,
            irreg_threshold=0.5,
            clustering_threshold=0.5
        )
        
        bitmask0 = np.array([True, True, True, True])
        bitmask1 = np.array([False, False, False, False])
        
        bitmasks = [bitmask0, bitmask1]
        labels = manager.partitionLabels(bitmasks)
        
        expected = np.array([0, 0, 0, 0])
        assert np.array_equal(labels, expected)

    def test_partition_labels_returns_correct_dtype(self, small_complete_graph):
        """partitionLabels should return int array."""
        manager = Manager(
            G=small_complete_graph,
            eps=0.5,
            irreg_vtx_threshold=0.5**5 / 90,
            dev_vtx_threshold=0.5,
            irreg_vtx_count_threshold=0.1,
            dev_threshold=0.1,
            irreg_threshold=0.5,
            clustering_threshold=0.5
        )
        
        bitmask0 = np.array([True, False])
        bitmask1 = np.array([False, True])
        
        labels = manager.partitionLabels([bitmask0, bitmask1])
        assert labels.dtype == np.int64 or labels.dtype == np.int32

    @pytest.mark.parametrize(
        ("bitmasks", "error_fragment"),
        [
            (
                [
                    np.array([True, False, False]),
                    np.array([False, False, True]),
                ],
                "uncovered=1",
            ),
            (
                [
                    np.array([True, True, False]),
                    np.array([False, True, True]),
                ],
                "overlapping=1",
            ),
        ],
    )
    def test_partition_labels_rejects_invalid_mask_families(
        self,
        small_complete_graph,
        bitmasks,
        error_fragment,
    ):
        manager = Manager(G=small_complete_graph, eps=0.5)

        with pytest.raises(ValueError, match=error_fragment):
            manager.partitionLabels(bitmasks)


class TestGlobalPartitionRefinement:
    """Tests for the disjoint global partition invariant."""

    def test_common_refinement_is_disjoint_and_exhaustive(self):
        parent = np.ones(6, dtype=bool)
        cuts = {
            0: [
                np.array([True, True, True, False, False, False]),
                np.array([True, False, False, True, False, False]),
            ]
        }

        refined = AlgorithmRunner._common_refinement([parent], cuts)
        membership = np.sum(np.asarray(refined, dtype=int), axis=0)

        assert len(refined) == 4
        assert np.array_equal(membership, np.ones(6, dtype=int))

    def test_refinement_rechecks_cartesian_product(
        self,
        small_complete_graph,
        temp_dirs,
        monkeypatch,
    ):
        partition_dir, graph_dir = temp_dirs
        runner = AlgorithmRunner(
            small_complete_graph,
            eps=0.25,
            max_depth=4,
            partition_dir=partition_dir,
            graph_dir=graph_dir,
        )
        edges_A, edges_B = runner._get_edge_lists()
        calls = []

        def evaluate(direction, mask_A, mask_B, incidence_A, incidence_B):
            calls.append((direction, mask_A.copy(), mask_B.copy()))
            pathweight = sum(
                sum(mask_A[index] for index, _ in incidence_A[vertex])
                * sum(mask_B[index] for index, _ in incidence_B[vertex])
                for vertex in runner.V2
            )
            stats = runner._base_stats(direction, pathweight, 0, 0.0)
            if len(calls) == 1:
                stats.failure_reason = "FAILED_DEVIATION_CHECK"
                return {
                    "stats": stats,
                    "cut_A": np.arange(len(edges_A)) % 2 == 0,
                    "cut_B": np.arange(len(edges_B)) % 2 == 0,
                    "failed": True,
                }
            stats.failure_reason = "PASSED_GAMMA_CHECK"
            return {
                "stats": stats,
                "cut_A": None,
                "cut_B": None,
                "failed": False,
            }

        monkeypatch.setattr(runner, "_evaluate_global_pair", evaluate)
        parts_A, parts_B = runner._run_global_refinement()

        assert len(parts_A) == 2
        assert len(parts_B) == 2
        assert len(calls) == 5
        assert len(calls[1:]) == len(parts_A) * len(parts_B)
        assert np.all(np.sum(np.asarray(parts_A, dtype=int), axis=0) == 1)
        assert np.all(np.sum(np.asarray(parts_B, dtype=int), axis=0) == 1)


class TestBitmaskDisjointness:
    """Tests for verifying that assembled partitions produce disjoint bitmasks."""

    def test_edge_partition_assembler_maps_local_masks(self, small_complete_graph):
        """Local neighbor masks should map to global E12 and E23 masks."""
        gm = GraphManager(small_complete_graph)
        assembler = EdgePartitionAssembler(gm)
        v = gm.getV2()[0]
        A, B = gm.makeLinkPartition(v)

        mask_A = np.array([True] + [False] * (len(A) - 1), dtype=bool)
        mask_B = np.array([True] + [False] * (len(B) - 1), dtype=bool)
        E12_contrib, E23_contrib = assembler.map_vertex_partition_to_edges(v, mask_A, mask_B, A, B)

        assert E12_contrib.dtype == bool
        assert E23_contrib.dtype == bool
        assert np.sum(E12_contrib) == 1
        assert np.sum(E23_contrib) == 1

    def test_assemble_partition_consistency(self, small_complete_graph):
        """assemble_partition should return consistent arrays."""
        partition_dir = tempfile.mkdtemp(prefix="test_assemble_")
        graph_dir = tempfile.mkdtemp(prefix="test_assemble_graphs_")
        
        try:
            manager = Manager(
                G=small_complete_graph,
                eps=0.5,
                irreg_vtx_threshold=0.5**5 / 90,
                dev_vtx_threshold=0.5,
                irreg_vtx_count_threshold=0.1,
                dev_threshold=0.1,
                irreg_threshold=0.5,
                clustering_threshold=0.5,
                partition_dir=partition_dir,
                graph_dir=graph_dir,
            )
            
            manager.V2 = manager.graph_manager.getV2()
            
            # Create some simple partitions
            for v in manager.V2:
                A, B = manager.graph_manager.makeLinkPartition(v)
                save_full_partition(manager.partition_manager, v, "test", A, B)
            
            # Assemble and check consistency
            part_A, part_B = manager.assemble_partition("test")
            
            assert isinstance(part_A, np.ndarray)
            assert isinstance(part_B, np.ndarray)
            assert len(part_A) > 0
            assert len(part_B) > 0
        
        finally:
            shutil.rmtree(partition_dir, ignore_errors=True)
            shutil.rmtree(graph_dir, ignore_errors=True)

    def test_manager_run_on_small_graph_with_temp_storage(self, temp_dirs):
        """Manager.run should produce labels and clean temporary storage."""
        partition_dir, graph_dir = temp_dirs
        manager = Manager(
            G=nx.complete_graph(3),
            eps=0.5,
            irreg_vtx_threshold=0.5**5 / 90,
            dev_vtx_threshold=0.5,
            irreg_vtx_count_threshold=0.1,
            dev_threshold=0.1,
            irreg_threshold=0.5,
            clustering_threshold=2.0,
            partition_dir=partition_dir,
            graph_dir=graph_dir,
            max_depth=5,
        )

        labels_A, labels_B = manager.run()

        assert isinstance(labels_A, np.ndarray)
        assert isinstance(labels_B, np.ndarray)
        assert len(labels_A) > 0
        assert len(labels_B) > 0
        assert os.listdir(partition_dir) == []
        assert os.listdir(graph_dir) == []


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_single_vertex_graph(self):
        """Manager should handle single vertex graph."""
        G = nx.complete_graph(1)
        manager = Manager(
            G=G,
            eps=0.5,
            irreg_vtx_threshold=0.5**5 / 90,
            dev_vtx_threshold=0.5,
            irreg_vtx_count_threshold=0.1,
            dev_threshold=0.1,
            irreg_threshold=0.5,
            clustering_threshold=0.5
        )
        
        assert manager.graph_manager is not None

    def test_two_vertex_graph(self):
        """Manager should handle two vertex graph."""
        G = nx.complete_graph(2)
        manager = Manager(
            G=G,
            eps=0.5,
            irreg_vtx_threshold=0.5**5 / 90,
            dev_vtx_threshold=0.5,
            irreg_vtx_count_threshold=0.1,
            dev_threshold=0.1,
            irreg_threshold=0.5,
            clustering_threshold=0.5
        )
        
        assert manager.graph_manager is not None

    def test_compute_direction_code_length_consistency(self, small_complete_graph):
        """Direction code length should be consistent across calls."""
        manager = Manager(
            G=small_complete_graph,
            eps=0.5,
            irreg_vtx_threshold=0.5**5 / 90,
            dev_vtx_threshold=0.5,
            irreg_vtx_count_threshold=0.1,
            dev_threshold=0.1,
            irreg_threshold=0.5,
            clustering_threshold=0.5
        )
        
        direction = "i0d1i2d3"
        length1 = manager.compute_direction_code_length(direction)
        length2 = manager.compute_direction_code_length(direction)
        
        assert length1 == length2 == 8


class TestHyperparameterVariations:
    """Tests with different epsilon values and hyperparameter functions."""

    def test_epsilon_half(self, small_complete_graph):
        """Test with epsilon = 1/2."""
        eps = 1/2
        manager = Manager(
            G=small_complete_graph,
            eps=eps,
            irreg_vtx_threshold=eps**5 / 90,
            dev_vtx_threshold=eps,
            irreg_vtx_count_threshold=0.1,
            dev_threshold=0.1,
            irreg_threshold=eps,
            clustering_threshold=eps,
            max_depth=10
        )
        
        assert manager.eps == eps
        assert manager.irreg_vtx_threshold == eps**5 / 90

    def test_epsilon_quarter(self, small_complete_graph):
        """Test with epsilon = 1/4."""
        eps = 1/4
        manager = Manager(
            G=small_complete_graph,
            eps=eps,
            irreg_vtx_threshold=eps**5 / 90,
            dev_vtx_threshold=eps,
            irreg_vtx_count_threshold=0.1,
            dev_threshold=0.1,
            irreg_threshold=eps,
            clustering_threshold=eps,
            max_depth=10
        )
        
        assert manager.eps == eps
        assert manager.irreg_vtx_threshold == eps**5 / 90
        # Smaller epsilon should give smaller thresholds
        assert manager.irreg_vtx_threshold < (1/2)**5 / 90

    def test_hyperparameter_scaling_with_epsilon(self, small_complete_graph):
        """Hyperparameters should scale appropriately with epsilon."""
        eps_values = [0.5, 0.25, 0.1]
        
        for eps in eps_values:
            manager = Manager(
                G=small_complete_graph,
                eps=eps,
                irreg_vtx_threshold=eps**5 / 90,
                dev_vtx_threshold=eps,
                irreg_vtx_count_threshold=0.1,
                dev_threshold=0.1,
                irreg_threshold=eps,
                clustering_threshold=eps
            )
            
            # Check that scaling is correct
            assert manager.irreg_vtx_threshold == eps**5 / 90
            assert manager.dev_vtx_threshold == eps
            assert manager.irreg_threshold == eps
            assert manager.clustering_threshold == eps


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
