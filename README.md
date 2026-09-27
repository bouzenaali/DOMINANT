# DOMINANT Reproduction

Reproducing Ding, Li, Bhanushali, Liu, "Deep Anomaly Detection on Attributed Networks" (SDM 2019),
as step one of a self-directed study of graph anomaly detection.

## Two versions, two purposes

**`dominant_from_scratch.py`** — DOMINANT implemented in raw NumPy: 3-layer GCN encoder, structure
decoder (`sigmoid(Z Z^T)`), attribute decoder (one more GCN layer), combined reconstruction loss,
and **manual backpropagation** (no autograd library). Includes a numerical gradient check that
verifies the hand-derived backprop math against finite differences before training starts. Runs
on a synthetic attributed network generated in the script itself (stochastic block model +
attribute prototypes), since it has no external dependencies and no internet requirement — built
this way specifically to force every formula in the paper to become real, verifiable code rather
than a library call.

Run: `python3 dominant_from_scratch.py`

Verified result (seed=42/7): gradient check passes (max relative error ~1e-8), and after 300
epochs of Adam training, ROC-AUC = **0.775** on the synthetic 300-node graph (12% injected
anomalies) — sitting right inside the paper's reported 0.75–0.82 range on BlogCatalog/Flickr/ACM.

**`dominant_pygod.py`** — the "official" reproduction, using the standard PyGOD library on
`inj_cora`, a real BOND-benchmark dataset (Cora citation network with the same style of injected
anomalies the DOMINANT paper itself uses). This is the version to actually cite in a report/repo,
since using an established library implementation on a standard benchmark is what a real
reproduction study is expected to look like — the from-scratch version is the "I understand every
line" companion piece, not the primary result.

Run (needs internet, on your own machine):
```
pip install -r requirements.txt
python3 dominant_pygod.py
```

## Anomaly injection method (used in the from-scratch version)

Matches the paper's Section 4.1:
- **Structural anomalies**: pick small groups of nodes and force them fully connected (artificial
  cliques) — anomalous because real social/citation graphs are sparse, so a tight clique stands out.
- **Attribute anomalies**: pick a node, sample k random candidate nodes, swap in the attributes of
  whichever candidate is most different (max Euclidean distance) — anomalous because the node's
  attributes now clash with its actual neighborhood.

## Findings — debugging the real reproduction on `inj_cora`

The synthetic from-scratch run gave a clean AUC of 0.775 on the first try, because the toy graph
was built simple enough not to expose any real failure modes. The real benchmark dataset
(`inj_cora`, via `dominant_colab.ipynb`) was a different story, and the debugging process along
the way surfaced more than the final number does on its own.

**Bug 1: the structure decoder was architecturally incapable of predicting "no edge."**
The first full run produced a completely flat loss (~0.100031, unmoving for 300 epochs) and an
alpha sweep where structure-only training (`alpha=0.0`) scored exactly at chance (AUC ≈ 0.50)
while attribute-only training (`alpha=1.0`) was clearly best — the opposite of the paper's own
finding of a mid-range sweet spot (0.4–0.8). A diagnostic on `A_hat.mean()` (0.50018, i.e.
`sigmoid(0)`) pointed to the cause: the encoder's final layer used ReLU, which forces every
embedding value to be ≥ 0. Since `A_hat = sigmoid(Z · Zᵀ)`, and the dot product of two
non-negative vectors is always ≥ 0, the model could mathematically never output a connection
probability below 0.5 — for any pair of nodes. On a graph that is 99.85% non-edges, that's fatal:
the model has no way to express "these two nodes are probably not connected."

**Fix attempt 1 (partial): removing the final-layer ReLU, following the original Graph
Autoencoder paper's design (Kipf & Welling), to let embeddings — and therefore dot products — go
negative.** This did not change the outcome: `A_hat.mean()` stayed at 0.50174, loss still
flatlined. The activation function wasn't the whole story — the deeper issue was that plain MSE
loss on a target that's 99.85% zeros gives the optimizer very little incentive to move away from
predicting a near-constant value for every pair, regardless of what values are technically
reachable.

**Bug 1, real fix: weighted loss for extreme class imbalance.** Switched the structure loss from
plain MSE to weighted binary cross-entropy, with `pos_weight` set to the ratio of non-edges to
edges (~662:1 on this graph) — the same recipe used in the original GAE implementation for
exactly this problem. This worked: `A_hat.mean()` dropped to 0.0094–0.0154 (much closer to the
true 0.00151), and the loss genuinely decreased across training (0.277 → 0.169) instead of being
frozen.

**Bug 2: a silent multi-class label bug in the evaluation, not the model.** After fixing the
structure decoder, overall AUC at `alpha=0.6` *dropped* to 0.62 and the alpha sweep got worse, not
better — a confusing result on its own. The cause turned out to be upstream of the model entirely:
`inj_cora`'s labels aren't binary. `np.unique(data.y, return_counts=True)` showed four distinct
values — `0` (normal, 2570 nodes), `1` (structural/clique anomaly, 68 nodes), `2`
(attribute/contextual anomaly, 68 nodes), `3` (both at once, 2 nodes) — standard BOND-benchmark
encoding. The evaluation code had been computing `binary_labels = (labels == 1)`, which only
counts the 68 structural-type anomalies as real positives and silently treats the other 70 real
anomalies (types 2 and 3) as normal. Every AUC and alpha-sweep number up to this point had been
computed against that incomplete ground truth. Also worth noting: the "Anomalies: 210 (7.8%)"
figure printed when the dataset first loads is itself wrong — `data.y.sum()` adds up the label
*codes* (0×2570 + 1×68 + 2×68 + 3×2 = 210), not a count of anomalous nodes. The real count is
68+68+2 = 138 nodes (5.1%).

**Fix: `binary_labels = (labels != 0)`** — count any injected anomaly type as a positive.
Re-evaluating the already-fixed model against the correct labels gave the final, trustworthy
result:

- **ROC-AUC: 0.7826** — inside the paper's own reported range (0.75–0.82 on
  BlogCatalog/Flickr/ACM).
- Precision@50/100/200: 0.180 / 0.280 / 0.220; Recall@50/100/200: 0.065 / 0.203 / 0.319.
- Alpha sweep: 0.0→0.678, 0.2→0.687, 0.4→0.701, 0.6→0.675, 0.8→0.677, **1.0→0.727 (best)**.
  Structure-only is now clearly above chance (a real signal, unlike the pre-fix 0.50), and the
  gap between structure-only and attribute-only closed substantially — much closer to the paper's
  framing that both signals matter, even though attributes alone still edge out slightly here.

**A finding that survived every fix, and is worth stating on its own:** a clean, isolated
comparison — true-normal nodes vs. true structural-only anomalies (label 0 vs. label 1
specifically, no contamination from the other anomaly types) — shows `struct_err` is
statistically indistinguishable between the two groups (27.07 vs. 27.06). The structure decoder,
even correctly trained, never learns to flag the artificially injected cliques on Cora
specifically. Its contribution to the overall AUC comes almost entirely from indirectly catching
the *attribute*-type anomalies instead: because the encoder blends attributes and topology
together before producing each node's embedding, a node with swapped-in attributes gets a
corrupted embedding purely from its bad attributes, which then also throws off the structure
decoder's guess about that node's connectivity — even though its real connections never changed.
A plausible reason the clique injection itself doesn't register: real citation networks already
contain naturally dense topical clusters, so an artificial clique may not look statistically
different from a legitimate one, in a way it likely would on a social-network graph like
BlogCatalog or Flickr (the paper's own datasets).

## Independent confirmation: `dominant_pygod.py` (standard library implementation)

Ran on `inj_cora` locally, no modifications needed:

- **ROC-AUC: 0.7859** — essentially identical to the hand-built version's 0.7826.
- Precision@50/100/200: 0.160 / 0.250 / 0.310; Recall@50/100/200: 0.058 / 0.181 / 0.449.

Two independent implementations (a hand-built PyTorch/PyTorch Geometric model vs. the standard
PyGOD library) converging on the same AUC is a stronger validation than either result alone —
it makes a leftover implementation-specific bug much less likely. Precision@200 is even slightly
better in the library version (0.310 vs. 0.220), consistent with PyGOD's `DOMINANT` already
handling the class-imbalance problem internally by default — i.e., the bug this reproduction had
to find and fix by hand (plain MSE failing on a 99.85%-sparse adjacency matrix) is evidently a
well-known enough issue that mature library implementations already guard against it.

### Read the [report](./dominant_report.pdf) for more details.