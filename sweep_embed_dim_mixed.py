"""
Sweep embed_dim in {32, 64, 128} for the mixed-graph GNN (gnn_gw.py, SBM + ER).
Dataset is generated ONCE and shared across all runs to ensure fair comparison.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from gnn_gw import (
    generate_dataset,
    GWEmbedder,
    train_epoch,
    evaluate,
    compute_gw_distance,
)

# ── Config ────────────────────────────────────────────────────────────────────
N_PAIRS_TRAIN = 3000
N_PAIRS_TEST  = 100
N_PAIRS_SHIFT = 100
EMBED_DIMS     = [32, 64, 128]
N_EPOCHS       = 100
BATCH_SIZE     = 32
LR             = 1e-4
GW_SCALE       = 10.0
DEVICE         = 'cpu'
# ─────────────────────────────────────────────────────────────────────────────


def run_one(embed_dim, train_graphs, train_pairs, train_gw,
            test_graphs, test_pairs, test_gw,
            shift_graphs, shift_pairs, shift_gw):
    print(f"\n{'='*60}")
    print(f"  embed_dim = {embed_dim}")
    print(f"{'='*60}")

    model = GWEmbedder(in_channels=2, hidden_channels=64,
                       embed_dim=embed_dim, n_layers=3).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    train_losses = []
    for epoch in range(1, N_EPOCHS + 1):
        loss = train_epoch(model, optimizer, train_graphs, train_pairs, train_gw,
                           batch_size=BATCH_SIZE, device=DEVICE)
        scheduler.step()
        train_losses.append(loss)
        if epoch % 20 == 0:
            _, pr, sr, mse = evaluate(model, test_graphs, test_pairs, test_gw, DEVICE)
            print(f"  Epoch {epoch:3d} | Loss {loss:.4f} | "
                  f"Test Pearson {pr:.4f} | MSE {mse:.6f}")

    _, test_pearson, test_spearman, test_mse = evaluate(
        model, test_graphs, test_pairs, test_gw, DEVICE)
    _, shift_pearson, shift_spearman, shift_mse = evaluate(
        model, shift_graphs, shift_pairs, shift_gw, DEVICE)

    print(f"\n  Final Test  — Pearson: {test_pearson:.4f}  MSE: {test_mse:.6f}")
    print(f"  Final Shift — Pearson: {shift_pearson:.4f}  MSE: {shift_mse:.6f}")

    # Time GNN inference over the first N_BENCH test pairs
    N_BENCH = min(20, len(test_pairs))
    gnn_times = []
    model.eval()
    with torch.no_grad():
        for i, j in test_pairs[:N_BENCH]:
            g1, g2 = test_graphs[i].to(DEVICE), test_graphs[j].to(DEVICE)
            b1 = torch.zeros(g1.num_nodes, dtype=torch.long, device=DEVICE)
            b2 = torch.zeros(g2.num_nodes, dtype=torch.long, device=DEVICE)
            t0 = time.perf_counter()
            emb1 = model(g1.x, g1.edge_index, b1)
            emb2 = model(g2.x, g2.edge_index, b2)
            _ = torch.norm(emb1 - emb2, dim=-1).item()
            gnn_times.append(time.perf_counter() - t0)

    return {
        'embed_dim':     embed_dim,
        'test_pearson':  test_pearson,
        'test_mse':      test_mse,
        'shift_pearson': shift_pearson,
        'shift_mse':     shift_mse,
        'train_losses':  train_losses,
        'gnn_mean_ms':   np.mean(gnn_times) * 1e3,
        'gnn_std_ms':    np.std(gnn_times)  * 1e3,
    }


def plot_results(results, out_path):
    dims   = [r['embed_dim']     for r in results]
    tp     = [r['test_pearson']  for r in results]
    tm     = [r['test_mse']      for r in results]
    sp     = [r['shift_pearson'] for r in results]
    sm     = [r['shift_mse']     for r in results]

    x      = np.arange(len(dims))
    width  = 0.35
    labels = [str(d) for d in dims]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Mixed GNN (SBM + ER)(n_pairs={N_PAIRS_TRAIN})",
                 fontsize=13, fontweight='bold')

    # Pearson r
    ax = axes[0]
    bars1 = ax.bar(x - width/2, tp, width, label='Test (in-dist)',  color='steelblue')
    bars2 = ax.bar(x + width/2, sp, width, label='Shift (out-dist)', color='coral')
    ax.set_title("Pearson r")
    ax.set_xlabel("embed_dim")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis='y', alpha=0.4)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

    # MSE
    ax = axes[1]
    bars3 = ax.bar(x - width/2, tm, width, label='Test (in-dist)',  color='steelblue')
    bars4 = ax.bar(x + width/2, sm, width, label='Shift (out-dist)', color='coral')
    ax.set_title("MSE")
    ax.set_xlabel("embed_dim")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.4)
    for bar in bars3:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9)
    for bar in bars4:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9)

    # Training curves
    ax = axes[2]
    colors = ['steelblue', 'darkorange', 'seagreen']
    for r, color in zip(results, colors):
        ax.plot(range(1, len(r['train_losses']) + 1), r['train_losses'],
                label=f"dim={r['embed_dim']}", color=color, linewidth=1.5)
    ax.set_title("Training Loss per Epoch (log scale)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, which='both', alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")


def main():
    print("=== Generating Training Data (shared across all embed_dim runs) ===")
    train_graphs, train_pairs, train_gw = generate_dataset(
        n_pairs=N_PAIRS_TRAIN, min_nodes=20, max_nodes=100, min_blocks=2, max_blocks=5)

    print("\n=== Generating Test Data ===")
    test_graphs, test_pairs, test_gw = generate_dataset(
        n_pairs=N_PAIRS_TEST, min_nodes=20, max_nodes=100, min_blocks=2, max_blocks=5)

    print("\n=== Generating Distribution Shift Data ===")
    shift_graphs, shift_pairs, shift_gw = generate_dataset(
        n_pairs=N_PAIRS_SHIFT, min_nodes=100, max_nodes=200, min_blocks=3, max_blocks=10)

    train_gw = train_gw * GW_SCALE
    test_gw  = test_gw  * GW_SCALE
    shift_gw = shift_gw * GW_SCALE

    results = []
    for d in EMBED_DIMS:
        res = run_one(d,
                      train_graphs, train_pairs, train_gw,
                      test_graphs,  test_pairs,  test_gw,
                      shift_graphs, shift_pairs, shift_gw)
        results.append(res)

    # ── OT timing (independent of embed_dim, computed once) ──
    def pyg_to_adj(g):
        n = g.num_nodes
        A = np.zeros((n, n))
        ei = g.edge_index.numpy()
        A[ei[0], ei[1]] = 1
        return A

    N_BENCH = min(20, len(test_pairs))
    ot_times = []
    for i, j in test_pairs[:N_BENCH]:
        A1, A2 = pyg_to_adj(test_graphs[i]), pyg_to_adj(test_graphs[j])
        t0 = time.perf_counter()
        compute_gw_distance(A1, A2)
        ot_times.append(time.perf_counter() - t0)
    ot_mean_ms = np.mean(ot_times) * 1e3
    ot_std_ms  = np.std(ot_times)  * 1e3

    print("\n\n=== SUMMARY ===")
    print(f"{'embed_dim':>10} {'Test Pearson':>14} {'Test MSE':>12} "
          f"{'Shift Pearson':>15} {'Shift MSE':>12}")
    for r in results:
        print(f"{r['embed_dim']:>10} {r['test_pearson']:>14.4f} {r['test_mse']:>12.6f} "
              f"{r['shift_pearson']:>15.4f} {r['shift_mse']:>12.6f}")

    print(f"\n=== Inference Speed  (averaged over {N_BENCH} test pairs) ===")
    print(f"{'embed_dim':>10} {'GNN (ms)':>14} {'GNN std':>10} {'OT exact (ms)':>15} {'OT std':>10} {'Speedup':>10}")
    for r in results:
        speedup = ot_mean_ms / r['gnn_mean_ms']
        print(f"{r['embed_dim']:>10} {r['gnn_mean_ms']:>14.3f} {r['gnn_std_ms']:>10.3f} "
              f"{ot_mean_ms:>15.3f} {ot_std_ms:>10.3f} {speedup:>9.1f}x")

    plot_results(results, "sweep_embed_dim_mixed.png")


if __name__ == "__main__":
    main()
