import argparse
import os
import time
import warnings

import numpy as np
import torch

from dataset import generate_dataset, save_dataset, load_dataset, compute_gw_distance
from model import GWEmbedder
from train import train_epoch, evaluate
from plot_results import make_plots

warnings.filterwarnings("ignore")

SEED = 0
GW_SCALE = 10.0

TRAIN_DATA_PATH = "data/train_data.pt"
CHECKPOINT_PATH = "checkpoints/gw_model.pt"
RESULTS_PATH = "results/run_results.pt"
PLOT_PATH = "results/gw_results.png"


def get_train_data(regen_data):
    if not regen_data and os.path.exists(TRAIN_DATA_PATH):
        print(f"\n=== Loading cached training data from {TRAIN_DATA_PATH} ===")
        return load_dataset(TRAIN_DATA_PATH)

    print("\n=== Generating Training Data ===")
    graphs, pairs, gw = generate_dataset(
        n_pairs=3000, min_nodes=20, max_nodes=100, min_blocks=2, max_blocks=5
    )
    save_dataset(TRAIN_DATA_PATH, graphs, pairs, gw)
    print(f"Cached training data to {TRAIN_DATA_PATH}")
    return graphs, pairs, gw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regen-data", action="store_true",
                        help="Regenerate training data instead of loading the cache")
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    for d in ("data", "checkpoints", "results"):
        os.makedirs(d, exist_ok=True)

    device = 'cpu'

    # ── Generate datasets ──
    train_graphs, train_pairs, train_gw = get_train_data(args.regen_data)

    print("\n=== Generating Test Data ===")
    test_graphs, test_pairs, test_gw = generate_dataset(
        n_pairs=100, min_nodes=20, max_nodes=100, min_blocks=2, max_blocks=5
    )

    print("\n=== Generating Distribution Shift Data (larger graphs) ===")
    shift_graphs, shift_pairs, shift_gw = generate_dataset(
        n_pairs=100, min_nodes=100, max_nodes=200, min_blocks=3, max_blocks=10
    )

    # Scale GW distances
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

    # ── Save checkpoint ──
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"\nModel saved to {CHECKPOINT_PATH}")

    # ── Save results bundle (lets you redo plots later without retraining) ──
    results = {
        "train_losses": train_losses,
        "test_pearsons": test_pearsons,
        "test_gw": test_gw,
        "test_preds": test_preds,
        "test_pearson": test_pearson,
        "test_spearman": test_spearman,
        "test_mse": test_mse,
        "shift_gw": shift_gw,
        "shift_preds": shift_preds,
        "shift_pearson": shift_pearson,
        "shift_spearman": shift_spearman,
        "shift_mse": shift_mse,
    }
    torch.save(results, RESULTS_PATH)
    print(f"Results saved to {RESULTS_PATH}")

    # ── Plots ──
    make_plots(results, PLOT_PATH)
    print(f"Plot saved to {PLOT_PATH}")


if __name__ == "__main__":
    main()
