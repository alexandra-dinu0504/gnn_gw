import time

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from dataset import compute_gw_distance


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


def _pyg_to_adj(g):
    n = g.num_nodes
    A = np.zeros((n, n))
    ei = g.edge_index.numpy()
    A[ei[0], ei[1]] = 1
    return A


def benchmark_inference(model, graphs, pairs, device='cpu', n_bench=20):
    """Compare GNN embedding distance inference time with exact GW computation time."""
    bench_pairs = pairs[:n_bench]

    model.eval()
    gnn_times = []
    with torch.no_grad():
        for i, j in bench_pairs:
            g1, g2 = graphs[i].to(device), graphs[j].to(device)
            b1 = torch.zeros(g1.num_nodes, dtype=torch.long, device=device)
            b2 = torch.zeros(g2.num_nodes, dtype=torch.long, device=device)
            t0 = time.perf_counter()
            emb1 = model(g1.x, g1.edge_index, b1)
            emb2 = model(g2.x, g2.edge_index, b2)
            _ = torch.norm(emb1 - emb2, dim=-1).item()
            gnn_times.append(time.perf_counter() - t0)

    ot_times = []
    for i, j in bench_pairs:
        A1, A2 = _pyg_to_adj(graphs[i]), _pyg_to_adj(graphs[j])
        t0 = time.perf_counter()
        compute_gw_distance(A1, A2)
        ot_times.append(time.perf_counter() - t0)

    gnn_mean, gnn_std = np.mean(gnn_times) * 1e3, np.std(gnn_times) * 1e3
    ot_mean,  ot_std  = np.mean(ot_times)  * 1e3, np.std(ot_times)  * 1e3
    return gnn_mean, gnn_std, ot_mean, ot_std