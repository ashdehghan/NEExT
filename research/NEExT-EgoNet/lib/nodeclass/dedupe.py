"""Feature-stage dedupe for k-hop bags.

At saturation (dense graphs, wide k) many egonets share an identical member
set — at full coverage, every bag IS the graph. Recomputing the per-bag
feature pass for each duplicate is the dominant cost of the local scope.

This helper computes node features once per unique member set and
replicates the resulting rows for every duplicate bag. Because egonets
number their internal nodes by sorted original id, identical member sets
produce identical feature blocks, so the replication is exact. The
embedding stage is NOT deduped on purpose: the approximate-Wasserstein
vectorizer fits a joint basis over all bags, so it must see the same bag
multiset as the naive path — feature-stage dedupe keeps the whole pipeline
bit-identical to the naive computation.
"""

from collections import defaultdict

import pandas as pd


def member_set_groups(egonets) -> dict:
    """Group egonet graph_ids by identical member sets.

    Returns {representative_graph_id: [graph_id, ...]} where each list
    contains every bag (including the representative) sharing one member set.
    """
    sig_to_rep = {}
    groups = defaultdict(list)
    for g in egonets.graphs:
        sig = frozenset(g.node_mapping.keys())
        rep = sig_to_rep.setdefault(sig, g.graph_id)
        groups[rep].append(g.graph_id)
    return dict(groups)


def dedupe_node_features(nxt, egonets, **feature_kwargs):
    """compute_node_features with duplicate bags computed once.

    Exact: returns the same Features content as calling
    nxt.compute_node_features(egonets, **feature_kwargs) directly, but the
    per-bag feature pass runs only on unique member sets.
    """
    groups = member_set_groups(egonets)
    n_dupes = len(egonets.graphs) - len(groups)
    if n_dupes == 0:
        return nxt.compute_node_features(egonets, **feature_kwargs)

    keep = set(groups)
    rep_collection = egonets.model_copy(
        update={
            "graphs": [g for g in egonets.graphs if g.graph_id in keep],
            "egonet_to_graph_node_mapping": {
                gid: pair for gid, pair in egonets.egonet_to_graph_node_mapping.items() if gid in keep
            },
        }
    )
    feats = nxt.compute_node_features(rep_collection, **feature_kwargs)
    fdf = feats.features_df

    parts = [fdf]
    for rep, gids in groups.items():
        block = fdf[fdf["graph_id"] == rep]
        for gid in gids:
            if gid != rep:
                parts.append(block.assign(graph_id=gid))
    full = pd.concat(parts, ignore_index=True).sort_values(["graph_id", "node_id"]).reset_index(drop=True)
    feats.features_df = full
    return feats


def khop_features_batched(
    nxt,
    source_collection,
    centers,
    label_column: str,
    k_hop: int,
    seed: int,
    feature_kwargs: dict,
    source_features=None,
    batch_size: int = 150,
):
    """Memory-bounded k-hop pipeline front half: bags + features in batches.

    Saturated-radius runs (dense graphs, wide k) OOM when all bags are
    materialized at once (~1,000 x 2,000-node subgraphs). This builds bags
    for `batch_size` centers at a time, computes their features (local via
    compute_node_features, global via project_source_features when
    `source_features` is given), reuses feature blocks across identical
    member sets via a cross-batch cache, and discards the bag objects.

    Returns (features, table): a Features object over ALL centers with
    globally unique graph_ids, and the matching bag table
    (graph_id, node_id, n_nodes, n_edges). The downstream Wasserstein
    embedding consumes only features_df, so no full bag collection is
    needed. Exact: identical member sets share identical internal node
    numbering (sorted original ids), so cached blocks equal recomputation.
    """
    from .bags import build_node_bags

    sig_cache = {}  # frozenset(members) -> (feature_block_without_graph_id, n_nodes, n_edges)
    feature_columns = None
    all_blocks, table_rows = [], []
    next_gid = 0
    for start in range(0, len(centers), batch_size):
        batch = centers[start:start + batch_size]
        bags = build_node_bags(nxt, source_collection, batch, label_column, method="k_hop", k_hop=k_hop, seed=seed)
        to_compute = []
        sigs = {}
        for g in bags.egonets.graphs:
            sigs[g.graph_id] = frozenset(g.node_mapping.keys())
            if sigs[g.graph_id] not in sig_cache:
                to_compute.append(g.graph_id)
        if to_compute:
            keep = set(to_compute)
            sub = bags.egonets.model_copy(
                update={
                    "graphs": [g for g in bags.egonets.graphs if g.graph_id in keep],
                    "egonet_to_graph_node_mapping": {
                        gid: pair for gid, pair in bags.egonets.egonet_to_graph_node_mapping.items() if gid in keep
                    },
                }
            )
            if source_features is not None:
                feats = sub.project_source_features(source_features)
            else:
                feats = nxt.compute_node_features(sub, **feature_kwargs)
            feature_columns = feats.feature_columns
            fdf = feats.features_df
            edge_counts = {g.graph_id: len(g.edges) for g in sub.graphs}
            node_counts = {g.graph_id: len(g.nodes) for g in sub.graphs}
            for gid in to_compute:
                block = fdf[fdf["graph_id"] == gid].drop(columns=["graph_id"])
                sig_cache[sigs[gid]] = (block, node_counts[gid], edge_counts[gid])
        table = bags.table.set_index("graph_id")
        for g in bags.egonets.graphs:
            block, n_nodes, n_edges = sig_cache[sigs[g.graph_id]]
            all_blocks.append(block.assign(graph_id=next_gid))
            table_rows.append(
                {"graph_id": next_gid, "node_id": table.loc[g.graph_id, "node_id"], "n_nodes": n_nodes, "n_edges": n_edges}
            )
            next_gid += 1
        del bags
    features_df = pd.concat(all_blocks, ignore_index=True)
    cols = ["graph_id"] + [c for c in features_df.columns if c != "graph_id"]
    features_df = features_df[cols].sort_values(["graph_id", "node_id"]).reset_index(drop=True)
    from NEExT.features import Features

    return Features(features_df, feature_columns), pd.DataFrame(table_rows)
