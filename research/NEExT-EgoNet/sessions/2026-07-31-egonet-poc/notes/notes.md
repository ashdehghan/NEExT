# Notes

<!-- Running lab notebook: dated entries, newest at the bottom. -->

## 2026-07-31

- Grounded in the codebase (3 exploration passes): egonet machinery
  (`EgonetCollection.compute_k_hop_egonets`), HF dataset catalog
  (`NEExT.workbench.dataset_library`, importable without the workbench extra), and the
  features→embeddings→ML pipeline. Key gotchas recorded in the session plan: quadratic
  attribute-copy in `_build_egonet`, unsorted egonet ids from the direct API,
  largest-component filter before center selection, unseeded `node_sample_rate`,
  re-entrancy bug if `compute_k_hop_egonets` is called twice on one collection.
- Built `code/datasets.py` (cached HF loader) + `code/run_experiment.py` (k sweep +
  baselines under identical split protocol).
- Ran the PoC: AIRPORTS_USA, k ∈ {1,2}. Results in `results.md`. Twist: center-node
  full-graph features (0.681) beat egonet embeddings (0.556 @ k=1, 0.626 @ k=2);
  k=2 took 6.7 h. Mechanism validated, comparison reframed the research question:
  find where local-subgraph signal beats/complements global node features.
- Feature-importance CSVs in `outputs/` (gitignored); worth a look next session before
  choosing the ablation set.
