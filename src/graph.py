"""Cross-source corroboration features, computed within each S1's own candidate group.

For every (query, candidate S1) pair that survives blocking + Stage-A pruning, look at
the OTHER queries that also list the same S1 as a candidate. run.py's own `context()`
already aggregates *predicted probabilities* over exactly this group (that is Stage C,
the "collective matcher"). This module adds a different, upstream signal on the SAME
group: raw embedding cosine similarity, computed once from the fine-tuned dense encoder,
before any LightGBM model has seen the pair. Two things fall out of it, computed per
member of the group (not one value shared by the whole group):

  * cross-source corroboration -- an S2 and an S3 query that are both strong candidates
    for one S1 AND look nearly identical in embedding space are close to independent
    confirmation that both are noisy copies of the same real business.
  * same-source crowding -- several S2 (or several S3) queries landing on one S1 with
    LOW mutual similarity is the sibling-decoy pattern the challenge data has by design
    (same core name, different real business, different house number): evidence against
    some of those pairs, not for them.

No new global search is needed: after Stage-A pruning, a candidate group (all queries
sharing one S1) is small (a handful on average), so this is pairwise cosine within
small groups, batched by group size so it stays a handful of GPU calls rather than one
per S1.
"""
import numpy as np
import polars as pl
import torch

GRAPH_COLS = ['sib_group_size', 'sib_cross_best_cos', 'sib_cross_count', 'sib_same_best_cos', 'sib_same_min_cos']

_SENTINEL = {
    'sib_cross_best_cos': -1.0, 'sib_cross_count': 0.0, 'sib_same_best_cos': -1.0, 'sib_same_min_cos': -1.0,
}


def _sentinel_frame(df):
    return df.select('qi', 'ri', pl.col('gsize').cast(pl.Int32).alias('sib_group_size'),
                     *[pl.lit(v, pl.Float32).alias(k) for k, v in _SENTINEL.items()])


def build_sib_features(pairs, qemb, max_group=40, high_cos=0.8, batch_elems=4_000_000, device=None):
    """pairs: qi, ri, src (one row per surviving candidate pair, any split). qemb: the
    full query embedding matrix for that split (float16, L2-normalised, row-indexed by
    qi -- exactly what `embed()` in run.py caches). Returns one row per input pair with
    the GRAPH_COLS columns (per-MEMBER, not per-group), safe to left-join back onto
    `pairs` by (qi, ri)."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    dim = qemb.shape[1]
    base = pairs.select('qi', 'ri', 'src').with_columns(pl.len().over('ri').alias('gsize'))

    parts = [_sentinel_frame(base.filter(pl.col('gsize') == 1))]
    huge = base.filter(pl.col('gsize') > max_group)
    if huge.height:
        parts.append(_sentinel_frame(huge))

    mid = base.filter((pl.col('gsize') > 1) & (pl.col('gsize') <= max_group))
    for gsize in sorted(mid['gsize'].unique().to_list()):
        sub = mid.filter(pl.col('gsize') == gsize).sort(['ri', 'qi'])
        n_groups = sub.height // gsize
        qi_flat = sub['qi'].to_numpy()          # already row-aligned with sub, length n_groups*gsize
        ri_flat = sub['ri'].to_numpy()
        qi_2d = qi_flat.reshape(n_groups, gsize)
        src_2d = sub['src'].to_numpy().reshape(n_groups, gsize)

        batch = max(1, batch_elems // (gsize * dim))
        cross_best = np.empty((n_groups, gsize), np.float32)
        cross_count = np.empty((n_groups, gsize), np.float32)
        same_best = np.empty((n_groups, gsize), np.float32)
        same_min = np.empty((n_groups, gsize), np.float32)

        for lo in range(0, n_groups, batch):
            hi = min(n_groups, lo + batch)
            idx = qi_2d[lo:hi]                                     # (b, gsize)
            v = torch.from_numpy(np.asarray(qemb[idx.reshape(-1)], dtype=np.float32)).to(device)
            v = v.view(hi - lo, gsize, dim)
            sim = torch.bmm(v, v.transpose(1, 2)).cpu().numpy()    # (b, gsize, gsize); sim[g,i,j]
            del v
            if device == 'cuda':
                torch.cuda.empty_cache()

            eye = np.eye(gsize, dtype=bool)[None]                  # exclude self
            same = (src_2d[lo:hi][:, :, None] == src_2d[lo:hi][:, None, :])
            cross_sim = np.where(eye | same, -2.0, sim)             # keep only cross-source, non-self
            same_sim = np.where(eye | ~same, -2.0, sim)             # keep only same-source, non-self

            # per-member (row i) stats, reduced over the "other member" axis (j) only
            cross_best[lo:hi] = np.where((cross_sim > -2.0).any(2), cross_sim.max(axis=2), -1.0)
            cross_count[lo:hi] = (cross_sim > high_cos).sum(axis=2).astype(np.float32)
            same_best[lo:hi] = np.where((same_sim > -2.0).any(2), same_sim.max(axis=2), -1.0)
            same_masked = np.where(same_sim > -2.0, same_sim, np.nan)
            with np.errstate(all='ignore'):
                row_min = np.nanmin(same_masked, axis=2)
            same_min[lo:hi] = np.where(np.isnan(row_min), -1.0, row_min)

        parts.append(pl.DataFrame({
            'qi': qi_flat.astype(np.uint32),
            'ri': ri_flat.astype(np.uint32),
            'sib_group_size': np.full(sub.height, gsize, dtype=np.int32),
            'sib_cross_best_cos': cross_best.reshape(-1),
            'sib_cross_count': cross_count.reshape(-1),
            'sib_same_best_cos': same_best.reshape(-1),
            'sib_same_min_cos': same_min.reshape(-1),
        }))

    return pl.concat(parts, how='diagonal_relaxed').select('qi', 'ri', *GRAPH_COLS)
