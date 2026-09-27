"""
DOMINANT reproduction using the standard PyGOD library, on a real BOND benchmark dataset
(inj_cora — Cora citation network with injected structural + attribute anomalies, same
injection methodology as the original DOMINANT paper, just packaged by the BOND authors
for consistent benchmarking).

Run this on your own machine (needs internet access to pip install + download the dataset —
the sandbox this was drafted in has no internet, so this script is UNTESTED by me; run it,
paste me the output, and we'll debug together if anything errors).

Setup (once):
    pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU is fine at this scale
    pip install torch_geometric
    pip install pygod

Then:
    python dominant_pygod.py
"""

import os

# PyGOD 1.1.0 loads its trusted Data checkpoint without specifying weights_only.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from pygod.detector import DOMINANT
from pygod.utils import load_data
from pygod.metric import eval_roc_auc, eval_precision_at_k, eval_recall_at_k

DATASET = "inj_cora"   # small BOND dataset, ~2700 nodes — fast even on a laptop CPU


def main():
    print(f"Loading {DATASET} ...")
    data = load_data(DATASET)
    print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}, "
          f"Anomalies: {int(data.y.sum())} ({100 * data.y.float().mean():.1f}%)")

    # weight=0.6 mirrors the paper's reasonable range (0.4-0.8); num_layers=4 in PyGOD
    # counts encoder+decoder layers together, roughly matching the paper's 3-layer encoder
    # + 1-layer attribute decoder.
    model = DOMINANT(hid_dim=64, num_layers=4, weight_decay=0.0, dropout=0.0,
                      lr=0.005, epoch=300, gpu=-1, weight=0.6, verbose=1)

    print("Training DOMINANT ...")
    model.fit(data)

    scores = model.decision_score_
    labels = data.y.bool()

    auc = eval_roc_auc(labels, scores)
    print(f"\nROC-AUC: {auc:.4f}   (paper's ballpark on similar datasets: ~0.75-0.82)")

    for k in (50, 100, 200):
        p = eval_precision_at_k(labels, scores, k)
        r = eval_recall_at_k(labels, scores, k)
        print(f"Precision@{k}: {p:.3f}   Recall@{k}: {r:.3f}")


if __name__ == "__main__":
    main()
