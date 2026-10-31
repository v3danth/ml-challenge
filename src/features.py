"""Pair features between a query record (S2/S3) and a candidate S1 record.

String kernels are compiled RapidFuzz pairwise calls (cpdist, all cores) or
Polars expressions; nothing loops over pairs in Python. No feature uses the
country label, because France appears only in test.
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist

from block import phonetic

VIEW_COLS = ['nm', 'core', 'sq', 'legal', 'f_domain', 'f_hash', 'f_dba', 'f_native', 'f_noaddr',
             'ad', 'street', 'loc', 'nums', 'hn']
STREET_WORDS_RE = (r'\b(?:st|rd|av|dr|ln|bd|ct|pl|cir|hwy|pkwy|trl|ter|sq|tpke|way|n|s|e|w|ne|nw|se|sw|ste|apt|fl|'
                   r'bldg|rm|po|box|r|ch|imp|all|rte|nr|opp|flat|plot|the|and|de|du|la|des|le|les|sec|twp|cnty)\b')

RETRIEVAL = ['cos', 'drank', 'lex', 'lrank', 'nkeys', 'in_dense', 'in_lex', 'cos_gap', 'lex_gap', 'cos_rank', 'n_cand']
STAGE_A = RETRIEVAL + ['a_core_ratio', 'a_core_tset', 'a_st_ratio', 'a_st_tset', 'a_sq_eq', 'a_hn_eq', 'a_hn_both', 'a_q_nostreet']


def _sim(a, b, scorer, workers):
    return cpdist(a, b, scorer=scorer, workers=workers, dtype=np.float32)


def retrieval_context(c):
    """c: qi, ri, cos, drank, lex, nkeys (null when a channel did not propose the pair)."""
    return c.with_columns(
        pl.col('drank').is_not_null().alias('in_dense'), pl.col('lex').is_not_null().alias('in_lex'),
        pl.col('drank').fill_null(99.0), pl.col('lex').fill_null(0.0), pl.col('nkeys').fill_null(0.0),
    ).with_columns(
        pl.col('lex').rank('ordinal', descending=True).over('qi').cast(pl.Float32).alias('lrank'),
        pl.col('cos').rank('ordinal', descending=True).over('qi').cast(pl.Float32).alias('cos_rank'),
        (pl.col('cos').max().over('qi') - pl.col('cos')).alias('cos_gap'),
        (pl.col('lex').max().over('qi') - pl.col('lex')).alias('lex_gap'),
        pl.len().over('qi').cast(pl.Float32).alias('n_cand'))


def stage_a_features(c, qv, rv, workers=-1):
    """Cheap features for the learned pruning stage (runs on the full candidate union)."""
    j = (c.join(qv.select('i', 'core', 'sq', 'street', 'hn').rename({'i': 'qi'}), on='qi', how='left')
         .join(rv.select('i', 'core', 'sq', 'street', 'hn').rename({'i': 'ri'}), on='ri', how='left', suffix='_r'))
    j = j.with_columns(
        pl.Series('a_core_ratio', _sim(j['core'].to_list(), j['core_r'].to_list(), fuzz.ratio, workers)),
        pl.Series('a_core_tset', _sim(j['core'].to_list(), j['core_r'].to_list(), fuzz.token_set_ratio, workers)),
        pl.Series('a_st_ratio', _sim(j['street'].to_list(), j['street_r'].to_list(), fuzz.ratio, workers)),
        pl.Series('a_st_tset', _sim(j['street'].to_list(), j['street_r'].to_list(), fuzz.token_set_ratio, workers)),
        (pl.col('sq') == pl.col('sq_r')).alias('a_sq_eq'),
        ((pl.col('hn') != '') & (pl.col('hn') == pl.col('hn_r'))).alias('a_hn_eq'),
        ((pl.col('hn') != '') & (pl.col('hn_r') != '')).alias('a_hn_both'),
        (pl.col('street') == '').alias('a_q_nostreet'))
    return j.drop('core', 'sq', 'street', 'hn', 'core_r', 'sq_r', 'street_r', 'hn_r')


PAIR = ['nm_ratio', 'core_ratio', 'core_tsort', 'core_tset', 'core_partial', 'core_jw', 'sq_ratio', 'sq_partial',
        'core_lev', 'st_ratio', 'st_tset', 'st_tsort', 'st_partial', 'ad_tset', 'ad_tsort', 'loc_tset', 'hn_lev',
        'hn_both', 'hn_eq', 'hn_sub', 'hn_reldiff', 'num_jacc', 'num_q_in_r', 'num_r_in_q', 'q_nnum', 'r_nnum',
        'st_word_jacc', 'st_word_q_in_r', 'core_jacc', 'core_q_in_r', 'core_r_in_q', 'q_ntok', 'r_ntok',
        'legal_eq', 'legal_conflict', 'q_nolegal', 'sq_q_in_r', 'sq_r_in_q', 'acronym',
        'f_domain', 'f_hash', 'f_dba', 'f_native', 'f_noaddr', 'q_nostreet', 'r_nostreet', 'q_noloc',
        'q_sqlen', 'r_sqlen', 'src3', 'core_wjacc', 'q_oov', 'q_maxidf', 'ph_ratio', 'ph_eq',
        'core_diff_n', 'core_wdiff', 'core_diff_minidf']


def token_idf(refs, base=None):
    """IDF of S1 core-name tokens in one catalogue; tokens unseen in S1 signal gibberish names.

    base: IDF table of another catalogue (train) that takes precedence for tokens it knows.
    Test is denser than train, so its own IDFs are systematically lower and the matcher sees
    a shifted feature; tokens only test has (French words) keep the test-side value."""
    n = refs.height
    t = refs.select(pl.col('core').str.extract_all(r'[a-z0-9]+').list.unique().alias('tok')).explode('tok').drop_nulls()
    own = t.group_by('tok').len().select('tok', (np.log(n) - pl.col('len').log()).cast(pl.Float32).alias('idf'))
    if base is None:
        return own
    return pl.concat([base, own.join(base, on='tok', how='anti')])


def pair_features(pairs, qv, rv, idf, workers=-1):
    """pairs: qi, ri (+ passthrough columns). qv/rv: views with local row id 'i'."""
    j = (pairs.join(qv.select('i', 'src', *VIEW_COLS).rename({'i': 'qi'}), on='qi', how='left')
         .join(rv.select('i', *VIEW_COLS).rename({'i': 'ri'}), on='ri', how='left', suffix='_r'))
    L = lambda col: j[col].to_list()
    nm, nmr, core, corer, sq, sqr = L('nm'), L('nm_r'), L('core'), L('core_r'), L('sq'), L('sq_r')
    st, str_, ad, adr, loc, locr, hn, hnr = L('street'), L('street_r'), L('ad'), L('ad_r'), L('loc'), L('loc_r'), L('hn'), L('hn_r')
    w = workers
    f = {
        'nm_ratio': _sim(nm, nmr, fuzz.ratio, w), 'core_ratio': _sim(core, corer, fuzz.ratio, w),
        'core_tsort': _sim(core, corer, fuzz.token_sort_ratio, w), 'core_tset': _sim(core, corer, fuzz.token_set_ratio, w),
        'core_partial': _sim(core, corer, fuzz.partial_ratio, w),
        'core_jw': _sim(core, corer, JaroWinkler.normalized_similarity, w),
        'sq_ratio': _sim(sq, sqr, fuzz.ratio, w), 'sq_partial': _sim(sq, sqr, fuzz.partial_ratio, w),
        'core_lev': _sim(core, corer, Levenshtein.distance, w),
        'st_ratio': _sim(st, str_, fuzz.ratio, w), 'st_tset': _sim(st, str_, fuzz.token_set_ratio, w),
        'st_tsort': _sim(st, str_, fuzz.token_sort_ratio, w), 'st_partial': _sim(st, str_, fuzz.partial_ratio, w),
        'ad_tset': _sim(ad, adr, fuzz.token_set_ratio, w), 'ad_tsort': _sim(ad, adr, fuzz.token_sort_ratio, w),
        'loc_tset': _sim(loc, locr, fuzz.token_set_ratio, w), 'hn_lev': _sim(hn, hnr, Levenshtein.distance, w),
    }
    ph = j.select(phonetic(pl.col('core')).alias('a'), phonetic(pl.col('core_r')).alias('b'))
    f['ph_ratio'] = _sim(ph['a'].to_list(), ph['b'].to_list(), fuzz.ratio, w)
    f['ph_eq'] = (ph['a'] == ph['b']).to_numpy()
    del nm, nmr, core, corer, sq, sqr, st, str_, ad, adr, loc, locr, hn, hnr, ph
    j = j.with_columns([pl.Series(k, v) for k, v in f.items()])

    def both(col):
        return (pl.col(col) != '') & (pl.col(col + '_r') != '')

    def jacc(a, b):
        return (a.list.set_intersection(b).list.len() / a.list.set_union(b).list.len().clip(1)).cast(pl.Float32)

    def contain(a, b):  # |a & b| / |a|
        return (a.list.set_intersection(b).list.len() / a.list.len().clip(1)).cast(pl.Float32)

    qn, rn = pl.col('nums').str.extract_all(r'\d+'), pl.col('nums_r').str.extract_all(r'\d+')
    qs = pl.col('street').str.replace_all(STREET_WORDS_RE, ' ').str.extract_all(r'[a-z]{2,}')
    rs = pl.col('street_r').str.replace_all(STREET_WORDS_RE, ' ').str.extract_all(r'[a-z]{2,}')
    qc, rc = pl.col('core').str.extract_all(r'[a-z0-9]+'), pl.col('core_r').str.extract_all(r'[a-z0-9]+')
    hq = pl.col('hn').str.slice(0, 15).cast(pl.Int64, strict=False)
    hr = pl.col('hn_r').str.slice(0, 15).cast(pl.Int64, strict=False)
    j = j.with_columns(
        both('hn').alias('hn_both'),
        (both('hn') & (pl.col('hn') == pl.col('hn_r'))).alias('hn_eq'),
        (both('hn') & (pl.col('hn_r').str.contains(pl.col('hn'), literal=True) |
                       pl.col('hn').str.contains(pl.col('hn_r'), literal=True))).alias('hn_sub'),
        pl.when(both('hn')).then((hq - hr).abs() / pl.max_horizontal(hq, hr).clip(1)).otherwise(-1)
        .fill_null(-1).cast(pl.Float32).alias('hn_reldiff'),
        pl.when(both('nums')).then(jacc(qn, rn)).otherwise(-1).alias('num_jacc'),
        pl.when(both('nums')).then(contain(qn, rn)).otherwise(-1).alias('num_q_in_r'),
        pl.when(both('nums')).then(contain(rn, qn)).otherwise(-1).alias('num_r_in_q'),
        qn.list.len().alias('q_nnum'), rn.list.len().alias('r_nnum'),
        pl.when(both('street')).then(jacc(qs, rs)).otherwise(-1).alias('st_word_jacc'),
        pl.when(both('street')).then(contain(qs, rs)).otherwise(-1).alias('st_word_q_in_r'),
        jacc(qc, rc).alias('core_jacc'), contain(qc, rc).alias('core_q_in_r'), contain(rc, qc).alias('core_r_in_q'),
        qc.list.len().alias('q_ntok'), rc.list.len().alias('r_ntok'),
        (pl.col('legal') == pl.col('legal_r')).alias('legal_eq'),
        (both('legal') & (pl.col('legal') != pl.col('legal_r'))).alias('legal_conflict'),
        (pl.col('legal') == '').alias('q_nolegal'),
        pl.col('sq_r').str.contains(pl.col('sq'), literal=True).alias('sq_q_in_r'),
        pl.col('sq').str.contains(pl.col('sq_r'), literal=True).alias('sq_r_in_q'),
        (rc.list.eval(pl.element().str.slice(0, 1)).list.join('') == pl.col('sq')).alias('acronym'),
        (pl.col('street') == '').alias('q_nostreet'), (pl.col('street_r') == '').alias('r_nostreet'),
        (pl.col('loc') == '').alias('q_noloc'),
        pl.col('sq').str.len_chars().alias('q_sqlen'), pl.col('sq_r').str.len_chars().alias('r_sqlen'),
        (pl.col('src') == 'S3').alias('src3'))
    # IDF-weighted name overlap, and gibberish evidence (query tokens unknown to S1).
    max_idf = float(idf['idf'].max()) if idf.height else 10.0
    tok = j.select(pl.int_range(pl.len(), dtype=pl.UInt32).alias('p'), qc.alias('qt'), rc.alias('rt')).with_columns(
        pl.col('qt').list.set_intersection(pl.col('rt')).alias('shared'), pl.col('qt').list.set_union(pl.col('rt')).alias('union'),
        pl.col('qt').list.set_symmetric_difference(pl.col('rt')).alias('diff'))
    idf_t = idf.rename({'tok': 't'})

    def weight(col):
        e = tok.select('p', pl.col(col).alias('t')).explode('t').drop_nulls().join(idf_t, on='t', how='left')
        return e.group_by('p').agg(pl.col('idf').fill_null(max_idf).sum().alias(col))

    wq = tok.select('p', pl.col('qt').alias('t')).explode('t').drop_nulls().join(idf_t, on='t', how='left')
    gib = wq.group_by('p').agg(pl.col('idf').is_null().mean().alias('q_oov'),
                               pl.col('idf').fill_null(max_idf).max().alias('q_maxidf'))
    # Lookalike evidence: a decoy keeps the name but swaps one word, usually a common category
    # word ("cafe" -> "bakery"), so few differing tokens with a LOW idf point to a different
    # business, while a differing rare token is more often a typo or an appended word.
    wd = (tok.select('p', pl.col('diff').alias('t')).explode('t').drop_nulls().join(idf_t, on='t', how='left')
          .group_by('p').agg(pl.len().alias('core_diff_n'), pl.col('idf').fill_null(max_idf).sum().alias('core_wdiff'),
                             pl.col('idf').fill_null(max_idf).min().alias('core_diff_minidf')))
    wt = (tok.select('p').join(weight('shared'), on='p', how='left').join(weight('union'), on='p', how='left')
          .join(gib, on='p', how='left').join(wd, on='p', how='left').sort('p'))
    j = j.with_columns(
        (wt['shared'].fill_null(0) / wt['union'].fill_null(0).clip(1e-6)).cast(pl.Float32).alias('core_wjacc'),
        wt['q_oov'].fill_null(0).cast(pl.Float32).alias('q_oov'),
        wt['q_maxidf'].fill_null(0).cast(pl.Float32).alias('q_maxidf'),
        wt['core_diff_n'].fill_null(0).cast(pl.Float32).alias('core_diff_n'),
        wt['core_wdiff'].fill_null(0).cast(pl.Float32).alias('core_wdiff'),
        wt['core_diff_minidf'].fill_null(-1).cast(pl.Float32).alias('core_diff_minidf'))
    keep = list(pairs.columns) + [c for c in ['src'] + PAIR if c not in pairs.columns]
    return j.select(keep)
