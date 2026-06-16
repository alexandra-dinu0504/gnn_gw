import torch
import matplotlib.pyplot as plt

RESULTS_PATH = "results/run_results.pt"
PLOT_PATH = "results/gw_results.png"


def make_plots(results, save_path):
    """Build the training-loss / test-fit / shift-fit figure from a results dict."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(results["train_losses"])
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_yscale('log')
    axes[0].grid(True, which='both')

    test_gw, test_preds = results["test_gw"], results["test_preds"]
    axes[1].scatter(test_gw, test_preds, alpha=0.5, s=20)
    mn, mx = min(test_gw.min(), test_preds.min()), max(test_gw.max(), test_preds.max())
    axes[1].plot([mn, mx], [mn, mx], 'r--', label='y=x')
    axes[1].set_title(f"Test Set\nPearson r={results['test_pearson']:.3f}")
    axes[1].set_xlabel("True GW Distance")
    axes[1].set_ylabel("Predicted GW Distance")
    axes[1].legend()
    axes[1].grid(True)

    shift_gw, shift_preds = results["shift_gw"], results["shift_preds"]
    axes[2].scatter(shift_gw, shift_preds, alpha=0.5, s=20, color='orange')
    mn, mx = min(shift_gw.min(), shift_preds.min()), max(shift_gw.max(), shift_preds.max())
    axes[2].plot([mn, mx], [mn, mx], 'r--', label='y=x')
    axes[2].set_title(f"Distribution Shift\nPearson r={results['shift_pearson']:.3f}")
    axes[2].set_xlabel("True GW Distance")
    axes[2].set_ylabel("Predicted GW Distance")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    results = torch.load(RESULTS_PATH, weights_only=False)
    make_plots(results, PLOT_PATH)
    print(f"Plot saved to {PLOT_PATH}")
