import argparse
import os
import warnings

import numpy as np
import torch

from dataset import generate_dataset, save_dataset, load_dataset
from model import GWEmbedder
from train import train_epoch, evaluate, benchmark_inference
from plot_results import make_plots

warnings.filterwarnings("ignore")

SEED = 0
GW_SCALE = 10.0
N_TRAIN_POOL = 20000  # size of the cached pool of training pairs

TRAIN_DATA_PATH = "data/train_data.pt"
CHECKPOINT_PATH = "checkpoints/gw_model.pt"
RESULTS_PATH = "results/run_results.pt"
PLOT_PATH = "results/gw_results.png"


def get_train_data(regen_data, n_pool):
    if not regen_data and os.path.exists(TRAIN_DATA_PATH):
        graphs, pairs, gw = load_dataset(TRAIN_DATA_PATH)
        if len(pairs) >= n_pool:
            print(f"\n=== Loading cached training data from {TRAIN_DATA_PATH} ({len(pairs)} pairs) ===")
            return graphs, pairs, gw
        print(f"\n=== Cached pool has only {len(pairs)} pairs, need {n_pool} — regenerating ===")

    print("\n=== Generating Training Data ===")
    graphs, pairs, gw = generate_dataset(
        n_pairs=n_pool, min_nodes=20, max_nodes=100, min_blocks=2, max_blocks=5
    )
    save_dataset(TRAIN_DATA_PATH, graphs, pairs, gw)
    print(f"Cached training data to {TRAIN_DATA_PATH}")
    return graphs, pairs, gw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regen-data", action="store_true",
                        help="Regenerate training data instead of loading the cache")
    parser.add_argument("--n-train", type=int, default=1000,
                        help="Number of training pairs to use this run (<= cached pool size)")
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    for d in ("data", "checkpoints", "results"):
        os.makedirs(d, exist_ok=True)

    device = 'cpu'

    # ── "Generate" datasets ──
    train_graphs, train_pairs, train_gw = get_train_data(args.regen_data, N_TRAIN_POOL)
    train_pairs, train_gw = train_pairs[:args.n_train], train_gw[:args.n_train]
    print(f"Training on {len(train_pairs)} of {N_TRAIN_POOL} cached pairs")

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
    model = GWEmbedder(in_channels=2, hidden_channels=64, embed_dim=32, n_layers=3).to(device)
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
    gnn_mean, gnn_std, ot_mean, ot_std = benchmark_inference(
        model, test_graphs, test_pairs, device=device, n_bench=N_BENCH
    )
    print(f"\n=== Inference Speed Comparison ({N_BENCH} pairs) ===")
    print(f"GNN inference:      {gnn_mean:8.3f} ± {gnn_std:.3f} ms")
    print(f"OT (exact GW):      {ot_mean:8.3f} ± {ot_std:.3f} ms")
    print(f"Speedup (OT/GNN):   {ot_mean / gnn_mean:.1f}x")

    # ── Save checkpoint ──
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"\nModel saved to {CHECKPOINT_PATH}")

    # ── Save results bundle (for generating plots without retraining) ──
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
