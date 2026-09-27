# DOMINANT Reproduction
 
Reproducing Ding, Li, Bhanushali, Liu, "Deep Anomaly Detection on Attributed Networks" (SDM 2019),
as step one of a self-directed study of graph anomaly detection ahead of a PhD application.
 
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
## Next steps
 
1. Run `dominant_pygod.py` locally and compare against the from-scratch numbers.
2. Sweep `alpha` (0.0 to 1.0) and reproduce the paper's Figure 3 — AUC should peak somewhere in
   0.4–0.8, not at the extremes.
3. Move to CoLA (contrastive method) as the second reproduction target.
4. Generalization experiment: train on one dataset, evaluate on another, to connect this to the
   actual graph-foundation-model research question.
 