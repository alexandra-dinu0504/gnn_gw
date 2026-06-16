import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr


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