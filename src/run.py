"""BER v8 end-to-end pipeline.

  prepared parquet ─► dense encoder fine-tuned on group E (20% of S1 owners)
      ─► for EVERY train query and EVERY test query, against the FULL catalogue of its split:
         lexical top-k ∪ dense top-k (per country)                       (candidate union)
      ─► Stage A: learned pruning (cheap features) ─────────────────────► candidate_pairs.tsv
      ─► sibling-graph features, when graph_enabled: embedding cosine within each S1's
         candidate group (cross-source corroboration + same-source crowding) -- see graph.py
      ─► cross-encoder score (fine-tuned E5-small, out-of-fold)
      ─► Stage B: pair matcher (LightGBM, ~80 features [+ sibling-graph features])
      ─► Stage C: collective matcher (Stage-B probabilities aggregated per query and per S1)
      ─► decode: one owner per query, prior-shift + expected-F0.5 per S1 ─► matching_results.tsv

Why the full catalogue: the hardest negatives are look-alikes (copies of an S1 with
another house number, and S1s sharing a name). Splitting the catalogue removes most
of them, which made v7's validation (0.991) far easier than the test (0.973).
Here every train query is scored against all 2.2M train S1s, all models are 2-fold
cross-fitted by owner group, and validation is the official macro-F0.5 over every
non-E S1 with predictions from *all* queries. Because test has a larger share of
unmatched queries than train, validation is also run on a "test-like" pool where
unmatched queries are resampled until the model-estimated matched share equals
test's; the decoding policy is chosen there. Test's matched share is estimated with
the EM prior-adjustment of Saerens, Latinne & Decaestecker (2002), not the raw mean of
posteriors, which is biased toward the training prior under label shift.
"""
import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from block import LexicalIndex  # noqa: E402
from decode import (apply_policy, assign, choose_policy, country_deltas, em_prior, macro_f05, prior_shift,  # noqa: E402
                    query_match_probability, shift_by_country)
from features import PAIR, STAGE_A, pair_features, retrieval_context, stage_a_features, token_idf  # noqa: E402
from graph import GRAPH_COLS, build_sib_features  # noqa: E402
from prep import PREP_VERSION, prepare_all  # noqa: E402

T0 = time.time()


def log(*a):
    print(f'[{(time.time() - T0) / 60:7.1f} min]', *a, flush=True)


def save_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, p)


def write_parquet(p, df):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp.parquet')
    df.write_parquet(tmp, compression='zstd')
    os.replace(tmp, p)


# ----------------------------------------------------------------------------- data and folds
def load_train(prep, cfg):
    """Groups by owner:
      -1  E1: trains the dense encoder
      -2  E2: trains the upstream scorers (Stage A pruning, cross-encoder); one model each,
          applied identically to every row, so their scores mean the same thing everywhere
       0/1 folds: Stages B and C are cross-fitted on them; validation uses their S1s.
    Unmatched queries are spread over E2 and the folds (the encoder only needs positives)."""
    s1 = pl.read_parquet(prep / 'train_s1.parquet').with_row_index('i')
    q = pl.read_parquet(prep / 'train_q.parquet').with_row_index('i')
    h = pl.col('entity_id').hash(seed=cfg['seed']).mod(10_000)
    e1, e2 = int(10_000 * cfg['encoder_share']), int(10_000 * (cfg['encoder_share'] + cfg['upstream_share']))
    h_f = pl.col('entity_id').hash(seed=cfg['seed'] + 1).mod(2).cast(pl.Int8)
    s1 = s1.with_columns(pl.when(h < e1).then(pl.lit(-1, pl.Int8)).when(h < e2).then(pl.lit(-2, pl.Int8))
                         .otherwise(h_f).alias('fold'))
    q = q.join(s1.select(pl.col('entity_id').alias('owner'), pl.col('i').alias('oi'), pl.col('fold').alias('ofold')),
               on='owner', how='left')
    h_u = pl.col('entity_id').hash(seed=cfg['seed'] + 2).mod(10_000) < int(10_000 * cfg['upstream_share'] / (1 - cfg['encoder_share']))
    q = q.with_columns(pl.when(pl.col('oi').is_not_null()).then(pl.col('ofold'))
                       .when(h_u).then(pl.lit(-2, pl.Int8)).otherwise(h_f).alias('fold')).drop('ofold')
    return s1, q.sort('i')


def load_test(prep):
    s = pl.read_parquet(prep / 'test_s1.parquet').with_row_index('i')
    q = pl.read_parquet(prep / 'test_q.parquet').with_row_index('i').with_columns(pl.lit(-1, pl.Int8).alias('fold'))
    return s, q


# ----------------------------------------------------------------------------- dense encoder
def encoder_options(cfg):
    """Pooling and input prefix must match how the base checkpoint was pre-trained:
    E5 mean-pools and expects "query: " on symmetric tasks; LaBSE uses the CLS token."""
    return {'max_len': cfg['encoder_max_len'], 'pooling': cfg.get('encoder_pooling', 'mean'),
            'prefix': cfg.get('encoder_prefix', '')}


def train_encoder(cfg, model_dir, work, s1, q):
    out = work / 'encoder'
    opts = json.dumps(encoder_options(cfg), sort_keys=True)
    if (out / 'DONE').exists():
        # A cached encoder (and its cached embeddings) is only valid for the pooling and
        # prefix it was trained with; mixing them silently degrades retrieval.
        cached = (out / 'DONE').read_text()
        if cached == 'ok':  # written before these options existed: mean pooling, no prefix
            cached = json.dumps({**encoder_options(cfg), 'pooling': 'mean', 'prefix': ''}, sort_keys=True)
        if cached != opts:
            raise SystemExit(f'{out} was trained with {cached}, config asks for {opts}; '
                             f'delete {out} and {work / "emb"} to retrain')
        return out
    from dense import Encoder, texts
    pos = q.filter((pl.col('fold') == -1) & pl.col('oi').is_not_null())
    pos = pos.sample(min(cfg['encoder_pairs'], pos.height), seed=cfg['seed'])
    log(f'encoder: {pos.height:,} training pairs (group E); hard negatives from the full catalogue')
    ix = LexicalIndex(s1, cfg['lexical_cap'])
    hard = []
    for lo in range(0, pos.height, cfg['query_chunk']):
        chunk = pos.slice(lo, cfg['query_chunk']).drop('i').with_row_index('i')
        r = ix.search(chunk, 5).join(chunk.select(pl.col('i').alias('qi'), 'oi'), on='qi').filter(pl.col('ri') != pl.col('oi'))
        hard.append(r.group_by('qi', maintain_order=True).first().select((pl.col('qi') + lo).alias('qi'), 'ri'))
    hard = pl.concat(hard)
    del ix
    gc.collect()
    rng = np.random.default_rng(cfg['seed'])
    neg = rng.integers(0, s1.height, pos.height).astype(np.int64)
    if hard.height:
        neg[hard['qi'].to_numpy()] = hard['ri'].to_numpy()
    st = texts(s1)
    enc = Encoder(model_dir, **encoder_options(cfg))
    enc.train(texts(pos), [st[i] for i in pos['oi'].to_list()], [st[i] for i in neg], epochs=cfg['encoder_epochs'],
              batch=cfg['encoder_batch'], lr=cfg['encoder_lr'], seed=cfg['seed'], log=log)
    enc.save(out)
    (out / 'DONE').write_text(opts)
    del enc
    gc.collect()
    return out


def embed(enc_dir, cfg, work, name, frame):
    p = work / 'emb' / f'{name}.npy'
    if p.exists():
        return np.load(p, mmap_mode='r')
    from dense import Encoder, texts
    enc = Encoder(enc_dir, **encoder_options(cfg))
    t = time.time()
    v = enc.encode(texts(frame), cfg['encode_batch'])
    log(f'embedded {name}: {len(v):,} records, {len(v) / max(1e-6, time.time() - t):,.0f}/s')
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p.with_suffix('.tmp.npy'), v)
    os.replace(p.with_suffix('.tmp.npy'), p)
    del enc
    gc.collect()
    return np.load(p, mmap_mode='r')


# ----------------------------------------------------------------------------- candidates
def exact_cos(qemb, remb, qi, ri, step=262_144):
    out = np.empty(len(qi), np.float32)
    for lo in range(0, len(qi), step):
        a = np.asarray(qemb[qi[lo:lo + step]], np.float32)
        b = np.asarray(remb[ri[lo:lo + step]], np.float32)
        out[lo:lo + step] = (a * b).sum(1)
    return out


def candidate_union(chunk, s1, qemb, remb, lex, cfg):
    """chunk: query views with global row 'i'. Returns the union with Stage-A features."""
    from dense import knn_candidates
    loc = chunk.with_columns(pl.col('i').alias('gi')).drop('i').with_row_index('i')
    lx = lex.search(loc, cfg['lexical_k'])
    dn = knn_candidates(loc, np.asarray(qemb[loc['gi'].to_numpy()]), s1, remb, cfg['dense_k'])
    u = dn.join(lx, on=['qi', 'ri'], how='full', coalesce=True)
    gi = loc['gi'].to_numpy()
    u = u.with_columns(pl.Series('qi', gi[u['qi'].to_numpy()]).cast(pl.UInt32))
    u = u.with_columns(pl.Series('cos', exact_cos(qemb, remb, u['qi'].to_numpy(), u['ri'].to_numpy())))
    if cfg.get('exclude_same_id'):
        # Matching a catalogue against itself (S2-x vs S1-x): a record's own entry is not a
        # candidate, and must be gone before retrieval_context ranks the rest.
        u = (u.join(chunk.select(pl.col('i').alias('qi'), pl.col('entity_id').str.slice(3).alias('_q')), on='qi')
             .join(s1.select(pl.col('i').alias('ri'), pl.col('entity_id').str.slice(3).alias('_r')), on='ri')
             .filter(pl.col('_q') != pl.col('_r')).drop('_q', '_r'))
    return stage_a_features(retrieval_context(u), chunk, s1, cfg['workers'])


def with_labels(c, chunk):
    if 'oi' not in chunk.columns:
        return c.with_columns(pl.lit(False).alias('y'), pl.lit(-1, pl.Int8).alias('fold'))
    return (c.join(chunk.select(pl.col('i').alias('qi'), 'oi', 'fold'), on='qi', how='left')
            .with_columns((pl.col('ri') == pl.col('oi')).fill_null(False).alias('y')).drop('oi'))


# ----------------------------------------------------------------------------- LightGBM helpers
def lgb_params(cfg, **over):
    p = dict(objective='binary', learning_rate=cfg['lgb_lr'], num_leaves=cfg['lgb_leaves'], min_data_in_leaf=200,
             feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255,
             num_threads=cfg['threads'], verbose=-1, seed=cfg['seed'])
    p.update(over)
    return p


def matrix(df, cols):
    return df.select(pl.col(cols).cast(pl.Float32)).to_numpy()


def early_stop_holdout(groups, seed, share=0.05):
    """All candidate rows of one query share its query-level features and context
    aggregates, so a row-level holdout leaks and stops boosting too late. Hold out
    whole queries instead (group-wise validation)."""
    return np.random.default_rng(seed).random(int(groups.max()) + 1)[groups] < share


def fit(X, y, w, cfg, rounds, params=None, name='model', groups=None):
    """groups: query id per row; when given, the early-stopping holdout is query-wise."""
    import lightgbm as lgb
    if groups is None:
        hold = np.random.default_rng(cfg['seed']).random(len(y)) < 0.05
    else:
        hold = early_stop_holdout(groups, cfg['seed'])
    dtr = lgb.Dataset(X[~hold], y[~hold], weight=None if w is None else w[~hold], free_raw_data=True)
    dva = lgb.Dataset(X[hold], y[hold], weight=None if w is None else w[hold], reference=dtr)
    b = lgb.train(params or lgb_params(cfg), dtr, rounds, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)])
    log(f'{name}: best iteration {b.best_iteration}, holdout {dict(b.best_score["valid_0"])}')
    return b


def save_models(models, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for k, m in enumerate(models):
        m.save_model(str(folder / f'{k}.txt'), num_iteration=m.best_iteration)
    (folder / 'DONE').write_text(str(len(models)))


def load_models(folder):
    import lightgbm as lgb
    if not (folder / 'DONE').exists():
        return None
    models = [lgb.Booster(model_file=str(folder / f'{k}.txt')) for k in range(int((folder / 'DONE').read_text()))]
    for m in models:
        m.best_iteration = m.current_iteration()
    return models


def crossfit_predict(models, X, fold):
    """Fold-f rows use model f (trained on the other fold); E and test rows average both."""
    out = np.empty(len(X), np.float32)
    for f in (0, 1):
        sel = fold == f
        if sel.any():
            out[sel] = models[f].predict(X[sel], num_iteration=models[f].best_iteration)
    rest = (fold != 0) & (fold != 1)
    if rest.any():
        out[rest] = np.mean([m.predict(X[rest], num_iteration=m.best_iteration) for m in models], axis=0)
    return out


def train_crossfit(X, y, fold, cfg, rounds, name, w=None, params=None, groups=None):
    models = []
    for f in (0, 1):
        tr = (fold != f) & (fold >= 0)
        models.append(fit(X[tr], y[tr], None if w is None else w[tr], cfg, rounds, params, f'{name} (trained on fold {1 - f})',
                          None if groups is None else groups[tr]))
    return models


def context(slim, p, prefix):
    """Aggregate a probability per query and per S1, excluding the pair itself."""
    def top2(g):
        return pl.when(pl.len().over(g) > 1).then(pl.col(p).top_k(2).min().over(g)).otherwise(0.0)

    strong = (pl.col(p) > 0.5).cast(pl.Float32)
    first_q = pl.col(p).rank('ordinal', descending=True).over('qi') == 1
    first_r = pl.col(p).rank('ordinal', descending=True).over('ri') == 1
    return slim.with_columns(
        pl.col(p).rank('ordinal', descending=True).over('qi').cast(pl.Float32).alias(f'{prefix}_rank_q'),
        pl.when(first_q).then(top2('qi')).otherwise(pl.col(p).max().over('qi')).cast(pl.Float32).alias(f'{prefix}_best_other_q'),
        pl.len().over('qi').cast(pl.Float32).alias(f'{prefix}_n_q'),
        (pl.col(p).sum().over('ri') - pl.col(p)).cast(pl.Float32).alias(f'{prefix}_s1_sum_other'),
        (strong.sum().over('ri') - strong).alias(f'{prefix}_s1_strong_other'),
        (strong.sum().over(['ri', 'src']) - strong).alias(f'{prefix}_s1_same_src_strong_other'),
        pl.when(first_r).then(top2('ri')).otherwise(pl.col(p).max().over('ri')).cast(pl.Float32).alias(f'{prefix}_s1_max_other'),
        pl.col(p).rank('ordinal', descending=True).over('ri').cast(pl.Float32).alias(f'{prefix}_s1_rank'),
        pl.len().over('ri').cast(pl.Float32).alias(f'{prefix}_s1_n'))


def ctx_cols(prefix):
    return [f'{prefix}_{c}' for c in ['rank_q', 'best_other_q', 'n_q', 's1_sum_other', 's1_strong_other',
                                      's1_same_src_strong_other', 's1_max_other', 's1_rank', 's1_n']]


def prune(df, tau, kmax):
    return df.filter((pl.col('pA') >= tau) & (pl.col('pA').rank('ordinal', descending=True).over('qi') <= kmax))


# ----------------------------------------------------------------------------- stage A
def stage_a(cfg, work, s1, q, qemb, remb, reports):
    """One pruning model, trained on group E2 queries against the full catalogue.
    The threshold is chosen on held-out E2 queries (1 in 5)."""
    models = load_models(work / 'models' / 'A')
    rep_path = reports / 'pruning.json'
    if models is not None and rep_path.exists():
        return models, json.loads(rep_path.read_text())['tau']
    rng = np.random.default_rng(cfg['seed'])
    pool = q.filter(pl.col('fold') == -2)
    sub = pool.sample(min(cfg['stageA_queries'], pool.height), seed=cfg['seed']).sort('i')
    folder = work / 'cand' / 'A'
    lex = None
    paths = []
    for lo in range(0, sub.height, cfg['query_chunk']):
        p = folder / f'{lo:010d}.parquet'
        paths.append(p)
        if p.exists():
            continue
        lex = lex or LexicalIndex(s1, cfg['lexical_cap'])
        chunk = sub.slice(lo, cfg['query_chunk'])
        write_parquet(p, with_labels(candidate_union(chunk, s1, qemb, remb, lex, cfg), chunk))
        log(f'Stage A sample (E2): candidates for {min(sub.height, lo + cfg["query_chunk"]):,}/{sub.height:,} queries')
    del lex

    def held(c, col='qi'):
        return (c[col].hash(seed=cfg['seed'] + 3) % 5 == 0).to_numpy()

    parts = []
    for p in paths:
        c = pl.read_parquet(p, columns=['qi', 'y'] + STAGE_A)
        keep = ~held(c) & (c['y'].to_numpy() | (rng.random(c.height) < cfg['stageA_neg_rate']))
        parts.append(c.filter(pl.Series(keep)))
    tr = pl.concat(parts)
    w = np.where(tr['y'].to_numpy(), 1.0, 1 / cfg['stageA_neg_rate'])
    log(f'Stage A training rows: {tr.height:,} ({int(tr["y"].sum()):,} positive)')
    models = [fit(matrix(tr, STAGE_A), tr['y'].to_numpy().astype(np.float32), w, cfg, cfg['stageA_rounds'],
                  lgb_params(cfg, learning_rate=0.1, num_leaves=127), 'Stage A', tr['qi'].to_numpy())]
    save_models(models, work / 'models' / 'A')
    del tr, parts
    scored = []
    for p in paths:
        c = pl.read_parquet(p, columns=['qi', 'ri', 'y'] + STAGE_A)
        c = c.filter(pl.Series(held(c)))
        scored.append(c.select('qi', 'ri', 'y', pl.Series('pA', predict_all(models, matrix(c, STAGE_A)))))
    scored = pl.concat(scored)
    held_q = sub.filter(pl.Series(held(sub, 'i')))
    n_true = int(held_q['oi'].is_not_null().sum())
    union_recall = scored['y'].sum() / n_true
    rows = []
    for tau in cfg['prune_tau_grid']:
        kept = prune(scored, tau, cfg['prune_max_per_query'])
        rows.append({'tau': tau, 'recall': kept['y'].sum() / n_true, 'pairs_per_query': kept.height / held_q.height})
    ok = [r for r in rows if r['recall'] >= union_recall - cfg['prune_max_recall_drop']]
    tau = max(ok, key=lambda r: r['tau'])['tau'] if ok else min(cfg['prune_tau_grid'])
    save_json(rep_path, {'heldout_queries': held_q.height, 'union_recall': union_recall,
                         'union_pairs_per_query': scored.height / held_q.height, 'grid': rows, 'tau': tau})
    return models, tau


def predict_all(models, X):
    return np.mean([m.predict(X, num_iteration=m.best_iteration) for m in models], axis=0).astype(np.float32)


def build_pruned(name, cfg, work, s1, q, qemb, remb, models_a, tau, idf):
    """Every query: candidate union -> Stage-A score -> prune -> full pair features."""
    folder = work / 'pruned' / name
    lex = None
    paths = []
    stats_path = folder / 'stats.json'
    for lo in range(0, q.height, cfg['query_chunk']):
        p = folder / f'{lo:010d}.parquet'
        paths.append(p)
        if p.exists():
            continue
        lex = lex or LexicalIndex(s1, cfg['lexical_cap'])
        chunk = q.slice(lo, cfg['query_chunk'])
        c = with_labels(candidate_union(chunk, s1, qemb, remb, lex, cfg), chunk)
        c = c.with_columns(pl.Series('pA', predict_all(models_a, matrix(c, STAGE_A))))
        k = prune(c, tau, cfg['prune_max_per_query'])
        part = {'union_hits': int(c.filter(pl.col('fold') >= 0)['y'].sum()),
                'pruned_hits': int(k.filter(pl.col('fold') >= 0)['y'].sum()),
                'union_pairs': c.height, 'pruned_pairs': k.height}
        write_parquet(p, pair_features(k, q, s1, idf, cfg['workers']))
        old = json.loads(stats_path.read_text()) if stats_path.exists() else {}
        old[p.name] = part
        save_json(stats_path, old)
        log(f'{name}: {min(q.height, lo + cfg["query_chunk"]):,}/{q.height:,} queries; kept {k.height:,} of {c.height:,} pairs')
    return paths


# ----------------------------------------------------------------------------- sibling graph
def build_graph_features(name, cfg, work, paths, qemb):
    """For every pruned pair, compute embedding-similarity features within its S1's
    candidate group (see graph.py), then write them out per chunk file for load_chunk.

    A group must span every query of the split: chunks are contiguous query ranges and
    queries are stored S2 first, then S3, so a per-chunk group is almost always single
    source and the cross-source features would be constant."""
    folder = work / 'graph' / name
    if not cfg.get('graph_enabled') or (folder / 'DONE').exists():
        return
    pairs = pl.concat([pl.read_parquet(p, columns=['qi', 'ri', 'src']) for p in paths]).with_columns(
        (pl.col('src') == 'S3').alias('src'))
    feats = build_sib_features(pairs, qemb, max_group=cfg['sib_max_group'], high_cos=cfg['sib_high_cos'])
    del pairs
    for p in paths:
        c = pl.read_parquet(p, columns=['qi', 'ri'])
        lo, hi = c['qi'].min(), c['qi'].max()
        write_parquet(folder / p.name, c.join(feats.filter(pl.col('qi').is_between(lo, hi)), on=['qi', 'ri'], how='left'))
    (folder / 'DONE').write_text('ok')
    log(f'sibling-graph features written for {name}: {feats.height:,} pairs, '
        f'{int((feats["sib_cross_best_cos"] > -1).sum()):,} with a cross-source sibling')


# ----------------------------------------------------------------------------- cross-encoder
def cross_encoder(cfg, work, ce_model, tr_paths, te_paths, s1_tr, q_tr, s1_te, q_te):
    """One cross-encoder fine-tuned on group-E2 pruned pairs; its logit is written for every pair
    with pA >= ce_min_pA (null below it). Reading both records jointly was the largest single
    gain reported by the 2026 leaderboard leaders; gating by pA keeps inference within a Kaggle
    session, since pairs Stage A already rules out cannot become matches anyway."""
    if not cfg['ce_enabled']:
        return
    import torch
    from cross import CrossEncoder, pair_texts
    tq, ts = pair_texts(q_tr), pair_texts(s1_tr)
    d = work / 'ce' / 'model'
    if not (d / 'DONE').exists():
        rows = pl.concat([pl.read_parquet(p, columns=['qi', 'ri', 'y', 'fold']) for p in tr_paths]).filter(pl.col('fold') == -2)
        neg, pos = rows.filter(~pl.col('y')), rows.filter(pl.col('y'))
        n_neg = min(neg.height, cfg['ce_train_pairs'] // 2)
        n_pos = min(pos.height, cfg['ce_train_pairs'] - n_neg)
        sel = pl.concat([pos.sample(n_pos, seed=cfg['seed']), neg.sample(n_neg, seed=cfg['seed'])])
        log(f'cross-encoder: training on {sel.height:,} E2 pairs ({n_pos:,} positive, {n_neg:,} hard negatives)')
        ce = CrossEncoder(ce_model, cfg['ce_max_len'])
        ce.fit([tq[i] for i in sel['qi'].to_list()], [ts[i] for i in sel['ri'].to_list()], sel['y'].to_numpy().astype(np.float32),
               epochs=cfg['ce_epochs'], batch=cfg['ce_batch'], lr=cfg['ce_lr'], seed=cfg['seed'], log=log)
        ce.save(d)
        (d / 'DONE').write_text('ok')
        del ce, rows, sel
        gc.collect()
        torch.cuda.empty_cache()

    def score(paths, qtexts, stexts, name):
        todo = [p for p in paths if not (work / 'ce' / name / p.name).exists()]
        if not todo:
            return
        model = CrossEncoder(d, cfg['ce_max_len'])
        scored = total = 0
        for p in todo:
            c = pl.read_parquet(p, columns=['qi', 'ri', 'pA'])
            sel = c.filter(pl.col('pA') >= cfg['ce_min_pA'])
            out = model.predict([qtexts[i] for i in sel['qi'].to_list()], [stexts[i] for i in sel['ri'].to_list()], cfg['ce_infer_batch'])
            ce = c.select('qi', 'ri').join(sel.select('qi', 'ri', pl.Series('ce', out)), on=['qi', 'ri'], how='left')
            write_parquet(work / 'ce' / name / p.name, ce)
            scored, total = scored + sel.height, total + c.height
        log(f'cross-encoder scored {name}: {scored:,} of {total:,} pairs (pA >= {cfg["ce_min_pA"]})')
        del model
        gc.collect()
        torch.cuda.empty_cache()

    score(tr_paths, tq, ts, 'train')
    del tq, ts
    score(te_paths, pair_texts(q_te), pair_texts(s1_te), 'test')


def load_chunk(p, work, name, cfg):
    c = pl.read_parquet(p)
    if cfg.get('graph_enabled'):
        c = c.join(pl.read_parquet(work / 'graph' / name / p.name), on=['qi', 'ri'], how='left')
    if cfg['ce_enabled']:
        # Chunks hold whole queries, so per-query rank and margin are exact here.
        c = c.join(pl.read_parquet(work / 'ce' / name / p.name), on=['qi', 'ri'], how='left').with_columns(
            pl.col('ce').rank('ordinal', descending=True).over('qi').cast(pl.Float32).alias('ce_rank_q'),
            (pl.col('ce').max().over('qi') - pl.col('ce')).alias('ce_gap_q'))
    return c


# ----------------------------------------------------------------------------- stages B and C
CE_COLS = ['ce', 'ce_rank_q', 'ce_gap_q']


FEATURE_GROUPS = {'graph': GRAPH_COLS, 'cross-encoder': CE_COLS, 'phonetic': ['ph_ratio', 'ph_eq'],
                  'lookalike': ['core_diff_n', 'core_wdiff', 'core_diff_minidf']}


def log_group_importance(models, cols, cfg):
    """Share of total split gain per added feature family (averaged over the cross-fit models)."""
    gain = np.mean([m.feature_importance('gain') for m in models], axis=0)
    share = {name: float(gain[[cols.index(c) for c in group if c in cols]].sum() / max(gain.sum(), 1e-12))
             for name, group in FEATURE_GROUPS.items() if any(c in cols for c in group)}
    log('Stage B gain share by feature family: ' + ', '.join(f'{k} {v:.1%}' for k, v in share.items()))


def matcher_cols(cfg):
    b = (STAGE_A + PAIR + ['pA'] + ctx_cols('a') + (GRAPH_COLS if cfg.get('graph_enabled') else [])
         + (CE_COLS if cfg['ce_enabled'] else []))
    return b, b + ['pB'] + ctx_cols('b')


def slim_frame(paths, cols):
    return pl.concat([pl.read_parquet(p, columns=cols) for p in paths])


def stages_bc(cfg, work, name, paths, models_b=None, models_c=None):
    """Scores (qi, ri, fold, y, pB, pC). Trains B and C (cross-fitted) when models are None."""
    B_COLS, C_COLS = matcher_cols(cfg)
    actx = context(slim_frame(paths, ['qi', 'ri', 'src', 'pA']), 'pA', 'a').drop('src', 'pA')
    if models_b is None:
        data = pl.concat([load_chunk(p, work, name, cfg).join(actx, on=['qi', 'ri'], how='left') for p in paths])
        X, y, fold = matrix(data, B_COLS), data['y'].to_numpy().astype(np.float32), data['fold'].to_numpy()
        qids = data['qi'].to_numpy()
        keys = data.select('qi', 'ri', 'src', 'fold', 'y')
        del data
        gc.collect()
        log(f'Stage B training rows: {int((fold >= 0).sum()):,} ({int(y[fold >= 0].sum()):,} positive)')
        models_b = train_crossfit(X, y, fold, cfg, cfg['matcher_rounds'], 'Stage B', groups=qids)
        save_models(models_b, work / 'models' / 'B')
        log_group_importance(models_b, B_COLS, cfg)
        keys = keys.with_columns(pl.Series('pB', crossfit_predict(models_b, X, fold)))
        del X
    else:
        keys = []
        for p in paths:
            c = load_chunk(p, work, name, cfg).join(actx, on=['qi', 'ri'], how='left')
            keys.append(c.select('qi', 'ri', 'src', 'fold', 'y', pl.Series('pB', crossfit_predict(models_b, matrix(c, B_COLS), c['fold'].to_numpy()))))
        keys = pl.concat(keys)
    bctx = context(keys.select('qi', 'ri', 'src', 'pB'), 'pB', 'b').drop('src')
    if models_c is None:
        data = pl.concat([load_chunk(p, work, name, cfg).join(actx, on=['qi', 'ri'], how='left')
                          .join(bctx, on=['qi', 'ri'], how='left') for p in paths])
        X, y, fold = matrix(data, C_COLS), data['y'].to_numpy().astype(np.float32), data['fold'].to_numpy()
        qids = data['qi'].to_numpy()
        order = data.select('qi', 'ri')
        del data
        gc.collect()
        models_c = train_crossfit(X, y, fold, cfg, cfg['matcher_rounds'], 'Stage C', groups=qids)
        save_models(models_c, work / 'models' / 'C')
        out = order.with_columns(pl.Series('pC', crossfit_predict(models_c, X, fold)))
        del X
    else:
        out = []
        for p in paths:
            c = load_chunk(p, work, name, cfg).join(actx, on=['qi', 'ri'], how='left').join(bctx, on=['qi', 'ri'], how='left')
            out.append(c.select('qi', 'ri', pl.Series('pC', crossfit_predict(models_c, matrix(c, C_COLS), c['fold'].to_numpy()))))
        out = pl.concat(out)
    return keys.join(out, on=['qi', 'ri']), models_b, models_c


# ----------------------------------------------------------------------------- validation
def validate_policy(cfg, scores, q, s1, pi_test, reports):
    """Official macro-F0.5 over every non-E S1, predictions from all train queries.
    'test-like': unmatched queries resampled so the model-estimated matched share of
    queries equals the one estimated on test."""
    eval_s1 = s1.filter(pl.col('fold') >= 0).select(pl.col('i').alias('ri'))
    truth = q.filter(pl.col('oi').is_not_null()).select(pl.col('i').alias('qi'), pl.col('oi').alias('ri')).join(eval_s1, on='ri')
    best = assign(scores.select('qi', 'ri', pl.col('pC').alias('p')))
    n_q = q.height
    est = float(best['p'].sum()) / n_q
    unmatched = best.join(q.filter(pl.col('oi').is_null()).select(pl.col('i').alias('qi')), on='qi')
    u = float(unmatched['p'].sum()) / max(1, q.filter(pl.col('oi').is_null()).height)
    n_unmatched = unmatched.height
    n_extra = int(max(0.0, (est * n_q - pi_test * n_q) / max(1e-6, pi_test - u))) if u < pi_test < est else 0
    n_extra = min(n_extra, 3 * n_unmatched)  # guard against a degenerate estimate
    extra = unmatched.sample(n_extra, with_replacement=True, seed=cfg['seed']) if n_extra else unmatched.head(0)
    extra = extra.with_columns((pl.int_range(pl.len(), dtype=pl.UInt32) + n_q).alias('qi'))
    pools = {'train_mix': best, 'test_like': pl.concat([best, extra.select(best.columns)])}
    restrict = lambda b: b.join(eval_s1, on='ri')
    log(f'validation: estimated matched share train {est:.4f} vs test {pi_test:.4f}; '
        f'test-like pool adds {n_extra:,} resampled unmatched queries')
    results = {}
    for name, pool in pools.items():
        pol, rep, grid = choose_policy(restrict(pool), truth, eval_s1.height, cfg['shift_grid'], log)
        results[name] = {'policy': pol, 'score': rep, 'grid': grid}
        log(f'VALIDATION [{name}] macro F0.5 = {rep["macro_f05"]:.5f} P {rep["precision"]:.5f} R {rep["recall"]:.5f} policy {pol}')
    chosen = results['test_like']['policy']
    country = {}
    pred = apply_policy(restrict(pools['train_mix']), chosen)
    for c in s1['country'].unique().to_list():
        ids = s1.filter((pl.col('country') == c) & (pl.col('fold') >= 0)).select(pl.col('i').alias('ri'))
        if ids.is_empty():  # a scope with no evaluation S1 has no macro F0.5
            continue
        country[c] = macro_f05(pred.join(ids, on='ri'), truth.join(ids, on='ri'), ids.height)
    stage_b = macro_f05(assign(scores.select('qi', 'ri', pl.col('pB').alias('p'))).filter(pl.col('p') >= 0.5)
                        .join(eval_s1, on='ri').select('qi', 'ri'), truth, eval_s1.height)
    rep = {'chosen_policy': chosen, 'train_mix': results['train_mix']['score'], 'test_like': results['test_like']['score'],
           'train_mix_best_policy': results['train_mix']['policy'], 'stage_b_only_threshold_0.5': stage_b,
           'by_country_train_mix': country, 'eval_s1': eval_s1.height, 'estimated_matched_share': {'train': est, 'test': pi_test},
           'grids': {k: v['grid'] for k, v in results.items()}}
    save_json(reports / 'policy.json', rep)
    return rep


# ----------------------------------------------------------------------------- main
def main(args, overrides=None):
    cfg = {**json.loads(Path(args.config).read_text()), **(overrides or {})}
    cfg['threads'] = cfg.get('threads') or os.cpu_count()
    cfg['workers'] = cfg['threads']
    work, out_dir, prep = Path(args.work), Path(args.out), Path(args.prepared)
    work.mkdir(parents=True, exist_ok=True)
    reports = work / 'reports'
    version_ok = (prep / 'PREP_VERSION').exists() and (prep / 'PREP_VERSION').read_text().strip() == PREP_VERSION
    if not version_ok:
        local = work / 'prepared'
        if (local / 'PREP_VERSION').exists() and (local / 'PREP_VERSION').read_text().strip() == PREP_VERSION:
            prep = local
        elif args.data:
            log(f'prepared parquet missing or not {PREP_VERSION}: rebuilding into {local}')
            prepare_all(args.data, local, log=log)
            prep = local
        else:
            raise SystemExit(f'{prep} is not prepared with {PREP_VERSION}; pass --data to rebuild')

    s1, q = load_train(prep, cfg)
    log(f'train: {s1.height:,} S1 ({int((s1["fold"] == -1).sum()):,} in E1, {int((s1["fold"] == -2).sum()):,} in E2), {q.height:,} queries')
    enc_dir = train_encoder(cfg, args.model, work, s1, q)
    remb = embed(enc_dir, cfg, work, 'train_s1', s1)
    qemb = embed(enc_dir, cfg, work, 'train_q', q)
    models_a, tau = stage_a(cfg, work, s1, q, qemb, remb, reports)
    pr = json.loads((reports / 'pruning.json').read_text())
    log(f'blocking (sample, full catalogue): union recall {pr["union_recall"]:.5f}, {pr["union_pairs_per_query"]:.1f} pairs/query; '
        f'tau={tau}: ' + str(next(r for r in pr['grid'] if r['tau'] == tau)))
    tr_paths = build_pruned('train', cfg, work, s1, q, qemb, remb, models_a, tau, token_idf(s1))
    build_graph_features('train', cfg, work, tr_paths, qemb)  # no-op unless graph_enabled
    del qemb, remb
    gc.collect()

    s1_t, q_t = load_test(prep)
    remb_t = embed(enc_dir, cfg, work, 'test_s1', s1_t)
    qemb_t = embed(enc_dir, cfg, work, 'test_q', q_t)
    te_idf = token_idf(s1_t, token_idf(s1) if cfg.get('idf_from_train') else None)
    te_paths = build_pruned('test', cfg, work, s1_t, q_t, qemb_t, remb_t, models_a, tau, te_idf)
    build_graph_features('test', cfg, work, te_paths, qemb_t)
    del qemb_t, remb_t
    gc.collect()

    cross_encoder(cfg, work, args.ce_model or args.model, tr_paths, te_paths, s1, q, s1_t, q_t)

    models_b, models_c = load_models(work / 'models' / 'B'), load_models(work / 'models' / 'C')
    b_cols, c_cols = matcher_cols(cfg)
    if models_b is None or models_c is None or (models_b[0].num_feature(), models_c[0].num_feature()) != (len(b_cols), len(c_cols)):
        if models_b is not None:
            log('cached Stage B/C models were trained on other features (graph/ce/phonetic changed): retraining')
        models_b = models_c = None
    tr_scores, models_b, models_c = stages_bc(cfg, work, 'train', tr_paths, models_b, models_c)
    te_scores, _, _ = stages_bc(cfg, work, 'test', te_paths, models_b, models_c)
    te_best = assign(te_scores.select('qi', 'ri', pl.col('pC').alias('p')))
    pi_train = float(assign(tr_scores.select('qi', 'ri', pl.col('pC').alias('p')))['p'].sum()) / q.height
    pi_test = em_prior(query_match_probability(te_best, q_t.height), pi_train)
    log(f'test matched share: plug-in {float(te_best["p"].sum()) / q_t.height:.4f}, EM {pi_test:.4f} (train {pi_train:.4f})')
    cfg['shift_grid'] = sorted(set(cfg['shift_grid']) | {round(prior_shift(pi_train, pi_test), 3)})

    stats = json.loads((work / 'pruned' / 'train' / 'stats.json').read_text())
    n_true = int(q.filter(pl.col('oi').is_not_null() & (pl.col('fold') >= 0)).height)
    blocking = {'union_recall_all_train': sum(v['union_hits'] for v in stats.values()) / n_true,
                'pruned_recall_all_train': sum(v['pruned_hits'] for v in stats.values()) / n_true,
                'union_pairs_per_query': sum(v['union_pairs'] for v in stats.values()) / q.height,
                'pruned_pairs_per_query': sum(v['pruned_pairs'] for v in stats.values()) / q.height}
    save_json(reports / 'blocking.json', blocking)
    log('blocking over all train queries: ' + json.dumps(blocking))
    rep = validate_policy(cfg, tr_scores, q, s1, pi_test, reports)
    if cfg.get('country_prior', True):
        deltas = country_deltas(te_best, q_t['country'].to_numpy(), pi_train, pi_test)
        save_json(reports / 'country_prior.json', deltas)
        log('per-country test calibration: ' + ', '.join(
            f'{c} share {v["matched_share"]:.4f} delta {v["delta"]:+.3f}' for c, v in deltas.items()))
        te_best = shift_by_country(te_best, q_t['country'].to_numpy(), deltas)
    pred = apply_policy(te_best, rep['chosen_policy'])
    cand = pl.concat([pl.read_parquet(p, columns=['qi', 'ri']) for p in te_paths])
    export(s1_t, q_t, cand, pred, out_dir, args.data, reports, rep, blocking)


def export(s1, q, cand, pred, out_dir, data, reports, rep, blocking):
    out_dir.mkdir(parents=True, exist_ok=True)
    qid = q.select(pl.col('i').alias('qi'), pl.col('entity_id').alias('qid'))

    def lists(pairs, col):
        g = pairs.join(qid, on='qi').group_by('ri').agg(pl.col('qid').unique().sort().str.join(',').alias(col))
        return s1.select(pl.col('i').alias('ri'), pl.col('entity_id').alias('source1_entity_id')).join(
            g, on='ri', how='left').sort('ri').select('source1_entity_id', pl.col(col).fill_null(''))

    m = lists(pred, 'matched_entity_ids')
    c = lists(pl.concat([cand, pred.select('qi', 'ri')]).unique(), 'candidate_entity_ids')
    mp, cp = out_dir / 'matching_results.tsv', out_dir / 'candidate_pairs.tsv'
    m.write_csv(mp, separator='\t', quote_style='never')
    c.write_csv(cp, separator='\t', quote_style='never')
    n = lambda col: pl.when(pl.col(col) == '').then(0).otherwise(pl.col(col).str.count_matches(',') + 1)
    sizes = c.select(n('candidate_entity_ids').alias('n'))['n']
    per_country = (m.join(c, on='source1_entity_id').join(
        s1.select(pl.col('entity_id').alias('source1_entity_id'), 'country'), on='source1_entity_id')
        .group_by('country').agg(n('matched_entity_ids').mean().alias('matches_per_s1'),
                                 (n('matched_entity_ids') == 0).mean().alias('empty_share'),
                                 n('candidate_entity_ids').mean().alias('candidates_per_s1')))
    stats = {'s1': s1.height, 'queries': q.height, 'candidate_links': int(sizes.sum()),
             'candidates_per_s1': {'mean': float(sizes.mean()), 'median': float(sizes.median()),
                                   'p90': float(sizes.quantile(0.9)), 'p99': float(sizes.quantile(0.99)), 'max': int(sizes.max())},
             'matched_links': pred.height, 'per_country': per_country.to_dicts(), 'blocking_train': blocking,
             'validation_test_like': rep['test_like'], 'validation_train_mix': rep['train_mix'], 'policy': rep['chosen_policy']}
    save_json(reports / 'test_summary.json', stats)
    shutil.copy(reports / 'test_summary.json', out_dir / 'run_summary.json')
    if data:
        from validate_outputs import validate
        validate(mp, cp, Path(data) / 'test', True)
    log(f'wrote {mp} and {cp}: {json.dumps(stats["candidates_per_s1"])} candidates/S1, {pred.height:,} links')


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepared', required=True, help='folder with prepared parquet (rebuilt under --work from --data if stale)')
    ap.add_argument('--model', required=True, help='multilingual-e5-small folder (MIT): dense encoder initialisation')
    ap.add_argument('--ce-model', default=None, help='cross-encoder initialisation (default: --model)')
    ap.add_argument('--work', required=True, help='scratch folder for caches')
    ap.add_argument('--out', required=True, help='folder for the two submission TSVs')
    ap.add_argument('--data', default=None, help='raw challenge data (train/, test/)')
    ap.add_argument('--config', default=str(Path(__file__).parents[1] / 'config.json'))
    return ap.parse_args()


if __name__ == '__main__':
    main(parse_args())
