"""Lexical candidate generation: an IDF-weighted inverted index over blocking keys.

Keys are country-scoped (a query and its owner always share a country in the
training data: 0 mismatches over 7.6M labelled pairs), hashed to u64, and keys
with more than `cap` S1 postings are dropped, so the cost per query is bounded
no matter how large the catalogue grows.

Key families (S = S1 side, Q = query side):
  N  core-name token          B  consecutive core-name token pair
  Q  whole squashed name      P  first 8 chars of squashed name (domains/hashtags)
  A  acronym: S1 initials == query single short token ("big massage" ~ "bm")
  S  street word              X  street number x street word
  Z  full street-number sequence (addresses made only of numbers)
  Y  house number x locality word (gibberish names with an intact address)
  L  core-name token x locality word (common names disambiguated by city)
  F  phonetic skeleton of the whole name ("tom cruz" ~ "tom cruise", "laxmi" ~ "lakshmi")
"""
import math

import polars as pl

STREET_WORDS = {'st', 'rd', 'av', 'dr', 'ln', 'bd', 'ct', 'pl', 'cir', 'hwy', 'pkwy', 'trl', 'ter', 'sq', 'tpke',
                'way', 'n', 's', 'e', 'w', 'ne', 'nw', 'se', 'sw', 'ste', 'apt', 'fl', 'bldg', 'rm', 'po', 'box',
                'r', 'ch', 'imp', 'all', 'rte', 'nr', 'opp', 'flat', 'plot', 'the', 'and', 'de', 'du', 'la',
                'des', 'le', 'les', 'sec', 'twp', 'cnty'}
VIEW = ['i', 'country', 'core', 'sq', 'street', 'loc']
# Ordered rewrites to one spelling per sound, in the spirit of Soundex / Double Metaphone,
# tuned for romanised Indic names as well as English and French ones.
PHONETIC = [('ph', 'f'), ('ck', 'k'), ('x', 'ks'), ('q', 'k'), ('z', 's'), ('c', 'k'), ('w', 'v'),
            (r'\Bh', ''), (r'\B[aeiouy]', '')]


def phonetic(expr):
    """Consonant skeleton of each token: spelling variants that sound alike map to one
    string, which edit distance alone misses ("cruz" vs "cruise" is 3 edits, both are "krs")."""
    for a, b in PHONETIC:
        expr = expr.str.replace_all(a, b)
    for ch in 'bdfgjklmnprstv':  # the regex engine has no backreferences, so collapse runs per letter
        expr = expr.str.replace_all(ch + '{2,}', ch)
    return expr


def _explode_key(frame, col, family):
    return (frame.select('i', 'country', col).explode(col).drop_nulls()
            .select('i', (pl.col('country') + '|' + family + '|' + pl.col(col)).alias('key')))


def keys(v, side):
    """v: normalised views with local row id 'i'. side: 'ref' (S1) or 'query'."""
    words = sorted(STREET_WORDS)
    name_tok = pl.col('core').str.split(' ').list.eval(pl.element().filter(pl.element().str.len_chars() >= 2)).list.unique()
    all_tok = pl.col('core').str.split(' ').list.eval(pl.element().filter(pl.element() != ''))
    bigram = all_tok.list.eval(pl.element() + '_' + pl.element().shift(-1)).list.eval(pl.element().drop_nulls())
    street_alpha = pl.col('street').str.extract_all(r'[a-z]{3,}').list.eval(
        pl.element().filter(~pl.element().is_in(words))).list.unique()
    street_num = pl.col('street').str.extract_all(r'\d+').list.unique(maintain_order=True)
    loc_tok = pl.col('loc').str.extract_all(r'[a-z]{3,}').list.unique()
    base = v.select('i', 'country', 'sq', name_tok.alias('nt'), bigram.alias('bg'), street_alpha.alias('sa'),
                    street_num.alias('sn'), loc_tok.alias('lt'),
                    all_tok.list.eval(pl.element().str.slice(0, 1)).list.join('').alias('acr'),
                    all_tok.list.len().alias('ntok'), phonetic(pl.col('core')).str.replace_all(' ', '').alias('ph'))
    c = pl.col('country') + '|'
    out = [_explode_key(base, 'nt', 'N'), _explode_key(base, 'bg', 'B'), _explode_key(base, 'sa', 'S')]
    out.append(base.filter(pl.col('sq').str.len_chars() >= 4).select('i', (c + 'Q|' + pl.col('sq')).alias('key')))
    out.append(base.filter(pl.col('sq').str.len_chars() >= 9).select('i', (c + 'P|' + pl.col('sq').str.slice(0, 8)).alias('key')))
    out.append(base.filter(pl.col('ph').str.len_chars() >= 4).select('i', (c + 'F|' + pl.col('ph')).alias('key')))
    if side == 'ref':
        out.append(base.filter(pl.col('ntok') >= 2).select('i', (c + 'A|' + pl.col('acr')).alias('key')))
    else:
        out.append(base.filter((pl.col('ntok') == 1) & pl.col('sq').str.len_chars().is_between(2, 6))
                   .select('i', (c + 'A|' + pl.col('sq')).alias('key')))
    x = base.select('i', 'country', pl.col('sn').list.head(2), 'sa').explode('sn').explode('sa').drop_nulls()
    out.append(x.select('i', (pl.col('country') + '|X|' + pl.col('sn') + '|' + pl.col('sa')).alias('key')))
    out.append(base.filter(pl.col('sn').list.len() >= 1).select('i', (c + 'Z|' + pl.col('sn').list.join('_')).alias('key')))
    y = base.select('i', 'country', pl.col('sn').list.head(1), 'lt').explode('sn').explode('lt').drop_nulls()
    out.append(y.select('i', (pl.col('country') + '|Y|' + pl.col('sn') + '|' + pl.col('lt')).alias('key')))
    nl = base.select('i', 'country', 'nt', 'lt').explode('nt').explode('lt').drop_nulls()
    out.append(nl.select('i', (pl.col('country') + '|L|' + pl.col('nt') + '|' + pl.col('lt')).alias('key')))
    return pl.concat(out).drop_nulls().select('i', pl.col('key').hash(seed=7).alias('key')).unique()


class LexicalIndex:
    def __init__(self, refs, cap=300):
        """refs: S1 views with local row id 'i' (UInt32)."""
        k = keys(refs.select(VIEW), 'ref')
        df = k.group_by('key').len()
        df = df.filter(pl.col('len') <= cap).with_columns(
            (pl.lit(math.log(max(2, refs.height))) - pl.col('len').log()).cast(pl.Float32).alias('idf'))
        self.post = k.join(df.select('key', 'idf'), on='key').rename({'i': 'ri'})

    def search(self, q, k):
        """q: query views with local row id 'i'. Returns qi, ri, lex, nkeys (top-k by lex)."""
        qk = keys(q.select(VIEW), 'query').rename({'i': 'qi'})
        hits = qk.join(self.post, on='key')
        sc = hits.group_by('qi', 'ri').agg(pl.col('idf').sum().alias('lex'), pl.len().cast(pl.Float32).alias('nkeys'))
        return (sc.sort(['qi', 'lex', 'ri'], descending=[False, True, False])
                .group_by('qi', maintain_order=True).head(k))
