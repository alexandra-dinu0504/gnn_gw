"""
Scaling experiment: how does GW prediction quality change with training set size?
Trains a fresh model for each of [1000, 3000, 10000] pairs and plots
Pearson r and MSE on both the test set and distribution-shift set.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from gnn_gw import (
    generate_dataset,
    GWEmbedder,
    train_epoch,
    evaluate,
)

GW_SCALE = 10.0
TRAIN_SIZES = [1000, 3000, 10000, 15000]
N_EPOCHS = 100


def train_and_evaluate(train_graphs, train_pairs, train_gw,
                       test_graphs, test_pairs, test_gw,
                       shift_graphs, shift_pairs, shift_gw,
                       device):
    model = GWEmbedder(in_channels=2, hidden_channels=64, embed_dim=128, n_layers=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    for epoch in range(1, N_EPOCHS + 1):
        train_epoch(model, optimizer, train_graphs, train_pairs, train_gw,
                    batch_size=32, device=device)
        scheduler.step()
        if epoch % 10 == 0:
            _, pr, sr, mse = evaluate(model, test_graphs, test_pairs, test_gw, device)
            print(f"  epoch {epoch:3d} | test Pearson: {pr:.4f} | MSE: {mse:.6f}")

    _, test_pearson, _, test_mse = evaluate(model, test_graphs, test_pairs, test_gw, device)
    _, shift_pearson, _, shift_mse = evaluate(model, shift_graphs, shift_pairs, shift_gw, device)
    return test_pearson, test_mse, shift_pearson, shift_mse


def main():
    device = 'cpu'

    print("=== Generating training pool (15 000 pairs) ===")
    train_graphs, train_pairs, train_gw = generate_dataset(
        n_pairs=15000, min_nodes=20, max_nodes=100, min_blocks=2, max_blocks=5
    )
    train_gw = train_gw * GW_SCALE

    print("\n=== Generating test set ===")
    test_graphs, test_pairs, test_gw = generate_dataset(
        n_pairs=100, min_nodes=20, max_nodes=100, min_blocks=2, max_blocks=5
    )
    test_gw = test_gw * GW_SCALE

    print("\n=== Generating distribution-shift set ===")
    shift_graphs, shift_pairs, shift_gw = generate_dataset(
        n_pairs=100, min_nodes=100, max_nodes=200, min_blocks=5, max_blocks=10
    )
    shift_gw = shift_gw * GW_SCALE

    results = {}
    for size in TRAIN_SIZES:
        print(f"\n=== Training on {size} pairs ===")
        tp, tm, sp, sm = train_and_evaluate(
            train_graphs, train_pairs[:size], train_gw[:size],
            test_graphs, test_pairs, test_gw,
            shift_graphs, shift_pairs, shift_gw,
            device,
        )
        results[size] = dict(test_pearson=tp, test_mse=tm,
                             shift_pearson=sp, shift_mse=sm)
        print(f"  Test  — Pearson: {tp:.4f}  MSE: {tm:.6f}")
        print(f"  Shift — Pearson: {sp:.4f}  MSE: {sm:.6f}")

    # ── Plot ──
    sizes          = TRAIN_SIZES
    test_pearsons  = [results[s]['test_pearson']  for s in sizes]
    shift_pearsons = [results[s]['shift_pearson'] for s in sizes]
    test_mses      = [results[s]['test_mse']      for s in sizes]
    shift_mses     = [results[s]['shift_mse']     for s in sizes]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(sizes, test_pearsons,  'o-',  label='Test set')
    axes[0].plot(sizes, shift_pearsons, 's--', label='Distribution shift')
    axes[0].set_xscale('log')
    axes[0].set_xticks(sizes)
    axes[0].set_xticklabels([str(s) for s in sizes])
    axes[0].set_title("Pearson r vs Training Size")
    axes[0].set_xlabel("Training pairs")
    axes[0].set_ylabel("Pearson r")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(sizes, test_mses,  'o-',  label='Test set')
    axes[1].plot(sizes, shift_mses, 's--', label='Distribution shift')
    axes[1].set_xscale('log')
    axes[1].set_xticks(sizes)
    axes[1].set_xticklabels([str(s) for s in sizes])
    axes[1].set_title("MSE vs Training Size")
    axes[1].set_xlabel("Training pairs")
    axes[1].set_ylabel("MSE")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("gw_scaling_gnn_gw.png", dpi=150)
    print("\nPlot saved to gw_scaling_gnn_gw.png")


if __name__ == "__main__":
    main()
