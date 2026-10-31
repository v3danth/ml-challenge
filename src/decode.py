"""Turn pair probabilities into per-S1 match lists, and score them exactly like the leaderboard.

1. Exclusivity: in the ground truth every S2/S3 record has at most one owner, so each
   query is only ever offered to its single most probable S1.
2. Per-S1 set selection maximising expected F0.5 (plug-in approximation):
       E[F(k)] ~ 1.25 * sum_{i<=k} p_i / (0.25 * (sum_all p + m) + k),   k >= 1
       E[F(0)] ~ prod_i (1 - p_i) * exp(-m)       (the singleton case scores 1.0)
   with m = expected number of true matches the candidates missed. Precision-heavy
   F0.5 and singletons are handled by one rule instead of a global threshold.
A plain probability threshold is also evaluated; the better policy on the held-out
universe is used for test.
"""
import numpy as np
import polars as pl


def assign(pairs):
    """pairs: qi, ri, p -> best S1 per query (ties broken by ri)."""
    return (pairs.sort(['qi', 'p', 'ri'], descending=[False, True, False])
            .group_by('qi', maintain_order=True).first())


def threshold_policy(best, t):
    return best.filter(pl.col('p') >= t).select('qi', 'ri')


def expected_f_policy(best, m=0.05, floor=0.05):
    b = best.filter(pl.col('p') >= floor).sort(['ri', 'p'], descending=[False, True])
    b = b.with_columns(
        pl.col('p').cum_sum().over('ri').alias('cum'),
        pl.int_range(1, pl.len() + 1).over('ri').alias('k'),
        pl.col('p').sum().over('ri').alias('tot'),
        (1 - pl.col('p')).clip(1e-9).log().sum().over('ri').alias('log_none'))
    b = b.with_columns((1.25 * pl.col('cum') / (0.25 * (pl.col('tot') + m) + pl.col('k'))).alias('ef'))
    b = b.with_columns(pl.col('ef').max().over('ri').alias('best_ef'),
                       (pl.col('log_none').exp() * np.exp(-m)).alias('ef0'))
    kstar = (b.filter(pl.col('ef') == pl.col('best_ef')).group_by('ri').agg(pl.col('k').min().alias('kstar')))
    b = b.join(kstar, on='ri')
    return b.filter((pl.col('best_ef') > pl.col('ef0')) & (pl.col('k') <= pl.col('kstar'))).select('qi', 'ri')


def macro_f05(pred, truth, n_s1):
    """pred, truth: frames (qi, ri) of predicted / true links; n_s1: catalogue size.
    Mean over *all* S1 entities (singletons score 1 if predicted empty, else 0)."""
    tp = pred.join(truth, on=['qi', 'ri']).group_by('ri').len().rename({'len': 'tp'})
    pc = pred.group_by('ri').len().rename({'len': 'np'})
    tc = truth.group_by('ri').len().rename({'len': 'nt'})
    s = tc.join(pc, on='ri', how='full', coalesce=True).join(tp, on='ri', how='left').fill_null(0)
    f = s.select(pl.when(pl.col('nt') == 0).then((pl.col('np') == 0).cast(pl.Float64))
                 .otherwise(1.25 * pl.col('tp') / (0.25 * pl.col('nt') + pl.col('np'))).alias('f'))
    untouched = n_s1 - s.height  # no truth and no prediction: singleton predicted empty = 1.0
    total_tp, total_p, total_t = int(tp['tp'].sum() or 0), pred.height, truth.height
    return {'macro_f05': float((f['f'].sum() + untouched) / n_s1),
            'precision': total_tp / max(1, total_p), 'recall': total_tp / max(1, total_t),
            'tp': total_tp, 'predicted': total_p, 'true': total_t}


def shift(best, delta):
    """Prior-shift correction in logit space (test has more unmatched queries than train)."""
    if delta == 0:
        return best
    z = (pl.col('p').clip(1e-7, 1 - 1e-7) / (1 - pl.col('p').clip(1e-7, 1 - 1e-7))).log() + delta
    return best.with_columns((1 / (1 + (-z).exp())).alias('p'))


def query_match_probability(best, n_queries):
    """Per-query probability of having any owner; queries with no surviving candidate get 0."""
    p = np.zeros(n_queries, np.float64)
    p[best['qi'].to_numpy()] = best['p'].to_numpy()
    return p


def em_prior(p, pi_train, iters=200, tol=1e-7):
    """Matched share of a new population from posteriors calibrated on the training prior.

    EM of Saerens, Latinne & Decaestecker (2002, Neural Computation 14(1)): re-weight each
    posterior by the ratio of new to old priors, re-estimate the prior as their mean, repeat.
    The plain mean of p is biased toward pi_train whenever the priors differ."""
    pi_train = float(np.clip(pi_train, 1e-6, 1 - 1e-6))
    pi = float(np.mean(p))
    for _ in range(iters):
        pos = pi / pi_train * p
        adj = pos / (pos + (1 - pi) / (1 - pi_train) * (1 - p))
        new = float(np.clip(adj.mean(), 1e-6, 1 - 1e-6))
        if abs(new - pi) < tol:
            return new
        pi = new
    return pi


def prior_shift(pi_old, pi_new):
    """Logit offset that moves posteriors from one class prior to another (the delta for `shift`)."""
    return float(np.log(pi_new / (1 - pi_new)) - np.log(pi_old / (1 - pi_old)))


def country_deltas(best, query_country, pi_train, pi_global, cap=1.5):
    """Per-country logit offset relative to the global test prior.

    The decoding policy is tuned for the pooled test prior; a country whose own EM-estimated
    matched share differs (France: no training labels, a different unmatched share) gets the
    extra offset. Capped because posteriors are least calibrated exactly where this matters.
    query_country: numpy array of country per query row id."""
    p = query_match_probability(best, len(query_country))
    out = {}
    for c in np.unique(query_country):
        sel = query_country == c
        pi_c = em_prior(p[sel], pi_train)
        out[str(c)] = {'queries': int(sel.sum()), 'matched_share': pi_c,
                       'delta': float(np.clip(prior_shift(pi_global, pi_c), -cap, cap))}
    return out


def shift_by_country(best, query_country, deltas):
    """Apply country_deltas offsets to (qi, ri, p) rows."""
    d = np.array([deltas[str(c)]['delta'] for c in query_country], np.float64)[best['qi'].to_numpy()]
    p = best['p'].to_numpy().clip(1e-7, 1 - 1e-7)
    return best.with_columns(pl.Series('p', 1 / (1 + np.exp(-(np.log(p / (1 - p)) + d)))))


def choose_policy(best, truth, n_s1, shifts=(0.0,), log=print):
    """Grid over prior shift x (threshold | expected-F0.5) on a validation pool."""
    results = []
    for d in shifts:
        b = shift(best, d)
        for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            results.append(({'kind': 'threshold', 't': t, 'shift': d}, macro_f05(threshold_policy(b, t), truth, n_s1)))
        for m in [0.0, 0.05, 0.1, 0.2]:
            for floor in [0.1, 0.3]:
                results.append(({'kind': 'expected_f', 'm': m, 'floor': floor, 'shift': d},
                                macro_f05(expected_f_policy(b, m, floor), truth, n_s1)))
    best_pol, best_r = max(results, key=lambda x: x[1]['macro_f05'])
    for pol, r in sorted(results, key=lambda x: -x[1]['macro_f05'])[:5]:
        log(f'  policy {pol}: macro F0.5 {r["macro_f05"]:.5f} P {r["precision"]:.5f} R {r["recall"]:.5f}')
    return best_pol, best_r, [{'policy': p, **r} for p, r in results]


def apply_policy(best, policy):
    b = shift(best, policy.get('shift', 0.0))
    if policy['kind'] == 'threshold':
        return threshold_policy(b, policy['t'])
    return expected_f_policy(b, policy['m'], policy['floor'])
