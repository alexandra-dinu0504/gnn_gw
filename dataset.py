import numpy as np
import torch
from torch_geometric.data import Data
import ot
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix


def generate_er_graph(n_nodes, p=None):
    """Erdos-Renyi random graph."""
    if p is None:
        p = np.random.uniform(0.05, 0.6)
    A = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if np.random.rand() < p:
                A[i, j] = A[j, i] = 1
    return A

def generate_sbm_graph(n_nodes, n_blocks, p_in=0.7, p_out=0.05):
    """Stochastic Block Model graph."""
    block_sizes = np.random.multinomial(n_nodes, np.ones(n_blocks) / n_blocks)
    block_sizes = np.maximum(block_sizes, 1)  # avoiding empty blocks

    # Adjacency matrix
    A = np.zeros((n_nodes, n_nodes))
    assignments = np.repeat(np.arange(n_blocks), block_sizes)

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            p = p_in if assignments[i] == assignments[j] else p_out
            if np.random.rand() < p:
                A[i, j] = 1
                A[j, i] = 1

    return A


def adj_to_pyg(A):
    """Convert adjacency matrix to PyG object."""
    n = A.shape[0]
    deg = A.sum(axis=1)  # (n,)

    A2 = A @ A
    a3_diag = np.sum(A * A2, axis=1)   # = diag(A @ A @ A)
    triads = deg * (deg - 1)
    cc = np.where(triads > 0, a3_diag / triads, 0.0)  # in [0, 1]

    deg_norm = deg / (deg.max() + 1e-8)
    x = torch.tensor(np.stack([deg_norm, cc], axis=1), dtype=torch.float)

    rows, cols = np.where(A > 0)
    edge_index = torch.tensor(np.stack([rows, cols]), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, num_nodes=n)


def _normalize_cost(C):
    """Replace infinities and normalize a shortest-path matrix to [0, 1]."""
    finite_pos = np.isfinite(C) & (C > 0)
    if not finite_pos.any():
        n = C.shape[0]
        C = np.ones((n, n))
        np.fill_diagonal(C, 0.0)
        return C
    max_finite = C[finite_pos].max()
    C = np.where(np.isinf(C), max_finite * 2, C) # replace inftys
    C = C / (C.max() + 1e-8) # normalise
    return C


def compute_gw_distance(A1, A2):
    n1, n2 = A1.shape[0], A2.shape[0]
    p = ot.unif(n1)
    q = ot.unif(n2)

    C1 = shortest_path(csr_matrix(A1), method='D', directed=False)
    C2 = shortest_path(csr_matrix(A2), method='D', directed=False)

    C1 = _normalize_cost(C1)
    C2 = _normalize_cost(C2)

    gw_dist = ot.gromov.gromov_wasserstein2(C1, C2, p, q, 'square_loss', verbose=False)
    return float(gw_dist)


def generate_dataset(n_pairs, min_nodes=20, max_nodes=100, min_blocks=2, max_blocks=5):
    """Generate a dataset of graph pairs with their GW distances."""
    graphs_pyg = []
    graphs_adj = []
    pairs = []
    gw_distances = []

    print(f"Generating {n_pairs} graph pairs (n={min_nodes}-{max_nodes} nodes)...")

    # Pre-generate a pool of graphs: mix types of structures
    n_graphs = int(n_pairs * 1.5)
    graph_types = ['sbm_dense', 'sbm_sparse', 'er_dense', 'er_sparse']
    for i in range(n_graphs):
        n = np.random.randint(min_nodes, max_nodes + 1)
        gtype = np.random.choice(graph_types)
        if gtype == 'sbm_dense':
            k = np.random.randint(min_blocks, max_blocks + 1)
            A = generate_sbm_graph(n, k, p_in=np.random.uniform(0.5, 0.9), p_out=np.random.uniform(0.01, 0.1))
        elif gtype == 'sbm_sparse':
            k = np.random.randint(min_blocks, max_blocks + 1)
            A = generate_sbm_graph(n, k, p_in=np.random.uniform(0.15, 0.4), p_out=np.random.uniform(0.005, 0.04))
        elif gtype == 'er_dense':
            A = generate_er_graph(n, p=np.random.uniform(0.3, 0.7))
        else: #gtype == 'er_sparse'
            A = generate_er_graph(n, p=np.random.uniform(0.04, 0.2))
        graphs_adj.append(A)
        graphs_pyg.append(adj_to_pyg(A))
        if (i + 1) % 10 == 0:
            print(f"  Generated {i+1}/{n_graphs} graphs", end="\r")

    print(f"\nComputing GW distances for {n_pairs} pairs...")
    idx = 0
    for _ in range(n_pairs):
        i, j = np.random.choice(n_graphs, 2, replace=False)
        gw = compute_gw_distance(graphs_adj[i], graphs_adj[j])
        pairs.append((i, j))
        gw_distances.append(gw)
        idx += 1
        if idx % 10 == 0:
            print(f"  Computed {idx}/{n_pairs} GW distances", end="\r")
    print(f"GW distance range: {min(gw_distances):.4f} - {max(gw_distances):.4f}, mean: {np.mean(gw_distances):.4f}")
    print(f"\nDone.")
    return graphs_pyg, pairs, np.array(gw_distances)


def save_dataset(path, graphs, pairs, gw_distances):
    """Cache a generated dataset to disk so it doesn't need to be recomputed."""
    torch.save({"graphs": graphs, "pairs": pairs, "gw_distances": gw_distances}, path)


def load_dataset(path):
    """Load a dataset previously saved with save_dataset."""
    d = torch.load(path, weights_only=False)
    return d["graphs"], d["pairs"], d["gw_distances"]