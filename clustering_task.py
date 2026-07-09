#!/usr/bin/env python3
from scipy.sparse import csr_matrix
import numpy as np


def _biadjacency_matrix(G, row_order, column_order):
    """Build a robust bipartite adjacency matrix for possibly empty cuts."""
    rows = list(row_order)
    cols = list(column_order)
    shape = (len(rows), len(cols))
    if not rows or not cols:
        return csr_matrix(shape, dtype=int)

    row_index = {node: idx for idx, node in enumerate(rows)}
    col_index = {node: idx for idx, node in enumerate(cols)}
    row_values = []
    col_values = []
    data = []

    for u, v, attrs in G.edges(data=True):
        if u in row_index and v in col_index:
            row_values.append(row_index[u])
            col_values.append(col_index[v])
        elif v in row_index and u in col_index:
            row_values.append(row_index[v])
            col_values.append(col_index[u])
        else:
            continue
        data.append(attrs.get("weight", 1))

    if not data:
        return csr_matrix(shape, dtype=int)
    return csr_matrix((data, (row_values, col_values)), shape=shape)


class Task:
    def __init__(
        self,
        link,
        partition,
        eps: float,
        irreg_vtx_threshold: float = None,
        dev_vtx_threshold: float = None,
        dev_split_threshold: float = None,
    ) -> None:
        self.link = self.G = link
        self.A, self.B = partition
        self.eps = eps
        self.irreg_vtx_threshold = eps**5 / 90 if irreg_vtx_threshold is None else irreg_vtx_threshold
        self.dev_vtx_threshold = eps if dev_vtx_threshold is None else dev_vtx_threshold
        self.dev_split_threshold = eps**5 if dev_split_threshold is None else dev_split_threshold

        self.M = _biadjacency_matrix(self.G, self.A, self.B)

        self.edges = int(self.M.sum())

        self.density = 0.0 if len(self.A) * len(self.B) == 0 else (
            self.edges / (len(self.A) * len(self.B))
        )

        self.deg_A_v = np.array(self.M.sum(axis=1)).ravel()
        self.deg_B_v = np.array(self.M.sum(axis=0)).ravel()
        self.pathweight = self.deg_A_v * len(self.B) if len(self.A) > 0 and len(self.B) > 0 else np.array([])

        self.deg_N_A = self.M.sum(axis=1) # degrees of N_A(v)
        self.deg_N_B = self.M.sum(axis=0) # degrees of N_B(v)

    def compute_local_deviation(self, gamma)  -> float:
        common_neighbors = self.M @ self.M.T
        if hasattr(common_neighbors, "toarray"):
            common_neighbors = common_neighbors.toarray()
        common_neighbors = np.asarray(common_neighbors)
        self.common_neighbor_matrix = common_neighbors

        spec = common_neighbors - (gamma**2) * len(self.B)
        self.spec_dev_matrix = spec
        return np.sum(spec)

    def compute_irregular_vertices(self, gamma) -> tuple[np.array, int]:
        deg_B_v = len(self.B)
        expected = gamma * deg_B_v
        threshold = self.irreg_vtx_threshold * deg_B_v

        positive = self.deg_A_v - expected > threshold
        negative = self.deg_A_v - expected < -threshold

        return positive, np.sum(positive), negative, np.sum(negative)

    def produce_new_masks(self, gamma)-> tuple[np.array, np.array]: 
        self.local_dev = self.compute_local_deviation(gamma)

        if len(self.A) == 0 or len(self.B) == 0:
            return np.zeros(len(self.A), dtype=bool), np.zeros(len(self.B), dtype=bool)

        deg_B_v = len(self.B)
        candidate_mask = (
            np.abs(self.deg_A_v - gamma * deg_B_v)
            < self.dev_vtx_threshold * deg_B_v
        )
        scores = np.sum(self.spec_dev_matrix, axis=1)
        candidate_indices = np.where(candidate_mask)[0]
        if candidate_indices.size == 0:
            candidate_indices = np.arange(len(self.A))
        u_star = candidate_indices[np.argmax(scores[candidate_indices])]

        common_neighbor_row = np.asarray(self.common_neighbor_matrix[u_star, :]).ravel()
        L_v = common_neighbor_row > self.dev_split_threshold * deg_B_v

        matrix_row = self.M[u_star:u_star + 1, :]
        if hasattr(matrix_row, "toarray"):
            matrix_row = matrix_row.toarray()
        mask_B = np.asarray(matrix_row).ravel() > 0

        return (L_v, mask_B)
        
