import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool
import ot
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
import time
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 1. DATASET GENERATION
# ──────────────────────────────────────────────
def generate_sbm_graph(n_nodes, n_blocks, p_in=0.7, p_out=0.05):
    """Stochastic Block Model graph."""
    block_sizes = np.random.multinomial(n_nodes, np.ones(n_blocks) / n_blocks)
    block_sizes = np.maximum(block_sizes, 1)  # avoid empty blocks

    # Build adjacency matrix
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
    C = np.where(np.isinf(C), max_finite * 2, C)
    C = C / (C.max() + 1e-8)
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

    # Pre-generate a pool of graphs — mix types for structural diversity
    n_graphs = int(n_pairs * 1.5)
    graph_types = ['sbm_dense', 'sbm_sparse']
    for i in range(n_graphs):
        n = np.random.randint(min_nodes, max_nodes + 1)
        gtype = np.random.choice(graph_types)
        if gtype == 'sbm_dense':
            k = np.random.randint(min_blocks, max_blocks + 1)
            A = generate_sbm_graph(n, k, p_in=np.random.uniform(0.5, 0.9), p_out=np.random.uniform(0.01, 0.1))
        else: #gtype == 'sbm_sparse':
            k = np.random.randint(min_blocks, max_blocks + 1)
            A = generate_sbm_graph(n, k, p_in=np.random.uniform(0.15, 0.4), p_out=np.random.uniform(0.005, 0.04))
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


# ──────────────────────────────────────────────
# 2. GNN ARCHITECTURE
# ──────────────────────────────────────────────

class GWEmbedder(nn.Module):
    """GCN that produces graph embedding."""

    def __init__(self, in_channels=1, hidden_channels=64, embed_dim=32, n_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.LayerNorm(hidden_channels))

        # Middle layers
        for _ in range(n_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.LayerNorm(hidden_channels))

        # Last conv layer
        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.bns.append(nn.LayerNorm(hidden_channels))

        # MLP for final embedding
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, embed_dim)
        )

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)

        # Mean and sum pooling
        x_mean = global_mean_pool(x, batch)
        x_sum = global_add_pool(x, batch)
        x = torch.cat([x_mean, x_sum], dim=1)

        return self.mlp(x)


# ──────────────────────────────────────────────
# 3. TRAINING
# ──────────────────────────────────────────────

def train_epoch(model, optimizer, graphs, pairs, gw_distances, batch_size=32, device='cpu'):
    model.train()
    total_loss = 0
    n = len(pairs)
    indices = np.random.permutation(n)

    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        if len(batch_idx) == 0:
            continue

        loss = torch.tensor(0.0, device=device)
        for k in batch_idx:
            i, j = pairs[k]
            gw_true = torch.tensor(gw_distances[k], dtype=torch.float, device=device)

            g1 = graphs[i].to(device)
            g2 = graphs[j].to(device)

            b1 = torch.zeros(g1.num_nodes, dtype=torch.long, device=device)
            b2 = torch.zeros(g2.num_nodes, dtype=torch.long, device=device)

            emb1 = model(g1.x, g1.edge_index, b1)
            emb2 = model(g2.x, g2.edge_index, b2)

            dist_pred = torch.norm(emb1 - emb2, dim=-1)
            loss = loss + (dist_pred - gw_true) ** 2

        loss = loss / len(batch_idx)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(1, n // batch_size)


@torch.no_grad()
def evaluate(model, graphs, pairs, gw_distances, device='cpu'):
    model.eval()
    preds = []
    for i, j in pairs:
        g1 = graphs[i].to(device)
        g2 = graphs[j].to(device)
        b1 = torch.zeros(g1.num_nodes, dtype=torch.long, device=device)
        b2 = torch.zeros(g2.num_nodes, dtype=torch.long, device=device)
        emb1 = model(g1.x, g1.edge_index, b1)
        emb2 = model(g2.x, g2.edge_index, b2)
        dist_pred = torch.norm(emb1 - emb2, dim=-1).item()
        preds.append(dist_pred)

    preds = np.array(preds)
    pearson_r, _ = pearsonr(preds, gw_distances)
    spearman_r, _ = spearmanr(preds, gw_distances)
    mse = np.mean((preds - gw_distances) ** 2)
    return preds, pearson_r, spearman_r, mse


# ──────────────────────────────────────────────
# 4. MAIN
# ──────────────────────────────────────────────

def main():
    device = 'cpu'

    # ── Generate datasets ──
    print("\n=== Generating Training Data ===")
    train_graphs, train_pairs, train_gw = generate_dataset(
        n_pairs=3000, min_nodes=20, max_nodes=100, min_blocks=2, max_blocks=5
    )

    print("\n=== Generating Test Data ===")
    test_graphs, test_pairs, test_gw = generate_dataset(
        n_pairs=100, min_nodes=20, max_nodes=100, min_blocks=2, max_blocks=5
    )

    print("\n=== Generating Distribution Shift Data (larger graphs) ===")
    shift_graphs, shift_pairs, shift_gw = generate_dataset(
        n_pairs=100, min_nodes=100, max_nodes=200, min_blocks=3, max_blocks=10
    )

    # Scale GW distances
    GW_SCALE = 10.0
    train_gw = train_gw * GW_SCALE
    test_gw  = test_gw  * GW_SCALE
    shift_gw = shift_gw * GW_SCALE

    # ── Model ──
    model = GWEmbedder(in_channels=2, hidden_channels=64, embed_dim=64, n_layers=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")

    # ── Training loop ──
    print("\n=== Training ===")
    n_epochs = 100
    train_losses = []
    test_pearsons = []

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(model, optimizer, train_graphs, train_pairs, train_gw,
                           batch_size=32, device=device)
        scheduler.step()
        train_losses.append(loss)

        if epoch % 10 == 0:
            _, pr, sr, mse = evaluate(model, test_graphs, test_pairs, test_gw, device)
            test_pearsons.append(pr)
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Test Pearson: {pr:.4f} | Spearman: {sr:.4f} | MSE: {mse:.6f}")
            #print({"\ntrain_gw": train_gw, "\ntest_gw": test_gw, "\nshift_gw": shift_gw})  # Debug print

    # ── Final evaluation ──
    print("\n=== Final Evaluation on Test Set ===")
    test_preds, test_pearson, test_spearman, test_mse = evaluate(
        model, test_graphs, test_pairs, test_gw, device
    )
    print(f"Pearson r:  {test_pearson:.4f}")
    print(f"Spearman r: {test_spearman:.4f}")
    print(f"MSE:        {test_mse:.6f}")

    print("\n=== Distribution Shift Evaluation ===")
    shift_preds, shift_pearson, shift_spearman, shift_mse = evaluate(
        model, shift_graphs, shift_pairs, shift_gw, device
    )
    print(f"Pearson r:  {shift_pearson:.4f}")
    print(f"Spearman r: {shift_spearman:.4f}")
    print(f"MSE:        {shift_mse:.6f}")

    # ── Inference speed comparison ──
    N_BENCH = min(20, len(test_pairs))
    bench_pairs = test_pairs[:N_BENCH]

    def pyg_to_adj(g):
        n = g.num_nodes
        A = np.zeros((n, n))
        ei = g.edge_index.numpy()
        A[ei[0], ei[1]] = 1
        return A

    model.eval()
    gnn_times = []
    with torch.no_grad():
        for i, j in bench_pairs:
            g1, g2 = test_graphs[i].to(device), test_graphs[j].to(device)
            b1 = torch.zeros(g1.num_nodes, dtype=torch.long, device=device)
            b2 = torch.zeros(g2.num_nodes, dtype=torch.long, device=device)
            t0 = time.perf_counter()
            emb1 = model(g1.x, g1.edge_index, b1)
            emb2 = model(g2.x, g2.edge_index, b2)
            _ = torch.norm(emb1 - emb2, dim=-1).item()
            gnn_times.append(time.perf_counter() - t0)

    ot_times = []
    for i, j in bench_pairs:
        A1, A2 = pyg_to_adj(test_graphs[i]), pyg_to_adj(test_graphs[j])
        t0 = time.perf_counter()
        compute_gw_distance(A1, A2)
        ot_times.append(time.perf_counter() - t0)

    gnn_mean, gnn_std = np.mean(gnn_times) * 1e3, np.std(gnn_times) * 1e3
    ot_mean,  ot_std  = np.mean(ot_times)  * 1e3, np.std(ot_times)  * 1e3
    print(f"\n=== Inference Speed Comparison ({N_BENCH} pairs) ===")
    print(f"GNN inference:      {gnn_mean:8.3f} ± {gnn_std:.3f} ms")
    print(f"OT (exact GW):      {ot_mean:8.3f} ± {ot_std:.3f} ms")
    print(f"Speedup (OT/GNN):   {ot_mean / gnn_mean:.1f}x")

    # ── Plots ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Training loss
    axes[0].plot(train_losses)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_yscale('log')
    axes[0].grid(True, which='both')

    # Test set: predicted vs true GW
    axes[1].scatter(test_gw, test_preds, alpha=0.5, s=20)
    mn, mx = min(test_gw.min(), test_preds.min()), max(test_gw.max(), test_preds.max())
    axes[1].plot([mn, mx], [mn, mx], 'r--', label='y=x')
    axes[1].set_title(f"Test Set\nPearson r={test_pearson:.3f}")
    axes[1].set_xlabel("True GW Distance")
    axes[1].set_ylabel("Predicted GW Distance")
    axes[1].legend()
    axes[1].grid(True)

    # Distribution shift: predicted vs true GW
    axes[2].scatter(shift_gw, shift_preds, alpha=0.5, s=20, color='orange')
    mn, mx = min(shift_gw.min(), shift_preds.min()), max(shift_gw.max(), shift_preds.max())
    axes[2].plot([mn, mx], [mn, mx], 'r--', label='y=x')
    axes[2].set_title(f"Distribution Shift\nPearson r={shift_pearson:.3f}")
    axes[2].set_xlabel("True GW Distance")
    axes[2].set_ylabel("Predicted GW Distance")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig("gw_results.png", dpi=150)
    print("\nPlot saved to gw_results.png")

    # Save model
    torch.save(model.state_dict(), "gw_model.pt")
    print("Model saved to gw_model.pt")


if __name__ == "__main__":
    main()