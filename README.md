## Repo layout

- `dataset.py` — random graph generators (SBM type), conversion to PyTorch
  Geometric `Data` objects, exact GW distance computation (with POT), and dataset
  generation/caching (pairs of graphs + their GW distance labels).
- `model.py` — `GWEmbedder`, a GCN + pooling network that maps a graph to a fixed-size
  embedding.
- `train.py` — the training loop, evaluation (Pearson/Spearman correlation + MSE against
  true GW distance), and a benchmark comparing GNN inference time to exact GW computation
  time.
- `plot_results.py` — builds the result figure (training loss curve, predicted-vs-true
  scatter plots for the test and distribution-shift sets). Can be re-run standalone from
  a saved results file, without retraining.
- `gnn_gw_sbm_only.py` — main entry point: generates/caches training data, trains the
  model, evaluates it, and saves a checkpoint, a results bundle, and the result plot.
- `old_gnn_gw_sbm_only.py` — the original cramped code, kept as a reference;
- `images/` — a few past result plots from earlier experiments.

## Usage

Train and test the model:

```bash
python gnn_gw_sbm_only.py
```

Outputs (all gitignored, not stored in the repo):

- `data/train_data.pt` — cached training pairs
- `checkpoints/gw_model.pt` — trained model weights
- `results/run_results.pt` — predictions, metrics, and losses from the run
- `results/gw_results.png` — the result figure

  Flags:

- `--n-train N` — train on `N` pairs sampled from the cached pool this run (default
  1000; must be `<= N_TRAIN_POOL=20 000`).
- `--regen-data` — force regenerating the training pool instead of loading the cache.
