"""Record normalisation.

Every record (S1, S2, S3) is turned into a set of canonical views that the
blocking and matching stages compare:

  name views     nm (normalised full name), core (legal forms / stop words removed),
                 sq (core without spaces, catches domains and hashtags), legal,
                 flags for domain / hashtag / "doing business as" / native script
  address views  street (tokens of the street-level components), loc (tokens of
                 frequent locality components: city, district, state),
                 nums (all numbers, leading zeros removed), hn (house number)

Transliteration of Indic scripts uses a token dictionary learned from the
training ground truth (native token -> Latin token of the aligned S1 name),
with AnyAscii as a fallback for unseen tokens. Everything is vectorised in
Polars except transliteration, which runs once per distinct non-ASCII string.
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl
from anyascii import anyascii

PREP_VERSION = 'v8.0'  # bump whenever normalisation changes; prepared parquet is checked against it
NATIVE = r'[ऀ-෿]'  # Devanagari .. Sinhala: every Indic script in the data

LEGAL = {
    'private': 'pvt', 'pvt': 'pvt', 'limited': 'ltd', 'ltd': 'ltd', 'incorporated': 'inc',
    'inc': 'inc', 'corporation': 'corp', 'corp': 'corp', 'company': 'co', 'co': 'co',
    'llc': 'llc', 'llp': 'llp', 'lp': 'lp', 'pllc': 'pllc', 'plc': 'plc', 'pc': 'pc',
    'sarl': 'sarl', 'sas': 'sas', 'sasu': 'sasu', 'eurl': 'eurl', 'sa': 'sa', 'sci': 'sci',
    'snc': 'snc', 'ei': 'ei', 'selarl': 'selarl', 'scop': 'scop', 'gmbh': 'gmbh',
}
LEGAL_FORMS = sorted(set(LEGAL.values()))
NAME_STOP = {'the', 'and', 'of', 'ms'}
# Dotted / spaced legal abbreviations collapse before tokenisation.
SPACED = [('public ltd', 'ltd'), ('public limited', 'ltd'), ('s a r l', 'sarl'), ('e u r l', 'eurl'), ('s a s u', 'sasu'), ('s a s', 'sas'),
          ('l l c', 'llc'), ('l l p', 'llp'), ('p l l c', 'pllc'), ('s c i', 'sci'),
          ('s n c', 'snc'), ('p v t', 'pvt'), ('i n c', 'inc'), ('l t d', 'ltd')]
ALIAS_PHRASE = (r'\b(?:doing business as|dba|formerly known as|formerly|fka|'
                r'also known as|aka|trading as)\b')

STREET = {
    'street': 'st', 'str': 'st', 'saint': 'st', 'road': 'rd', 'avenue': 'av', 'ave': 'av',
    'aven': 'av', 'drive': 'dr', 'drv': 'dr', 'lane': 'ln', 'boulevard': 'bd', 'blvd': 'bd',
    'boul': 'bd', 'court': 'ct', 'place': 'pl', 'circle': 'cir', 'highway': 'hwy',
    'parkway': 'pkwy', 'trail': 'trl', 'terrace': 'ter', 'square': 'sq', 'turnpike': 'tpke',
    'expressway': 'expy', 'freeway': 'fwy', 'point': 'pt', 'mount': 'mt', 'north': 'n',
    'south': 's', 'east': 'e', 'west': 'w', 'northeast': 'ne', 'northwest': 'nw',
    'southeast': 'se', 'southwest': 'sw', 'suite': 'ste', 'apartment': 'apt', 'floor': 'fl',
    'flr': 'fl', 'building': 'bldg', 'room': 'rm', 'township': 'twp', 'county': 'cnty',
    'rue': 'r', 'chemin': 'ch', 'chem': 'ch', 'impasse': 'imp', 'allee': 'all', 'route': 'rte',
    'quai': 'qu', 'cours': 'crs', 'faubourg': 'fbg', 'residence': 'res', 'lotissement': 'lot',
    'sector': 'sec', 'near': 'nr', 'opposite': 'opp', 'number': 'no', 'nos': 'no',
}
ADDR_NOISE = {'no', 'hno', 'dno', 'h', 'd', 'door', 'null', 'na', 'none', 'unit', 'c', 'o'}
US_STATES = {
    'alabama': 'al', 'alaska': 'ak', 'arizona': 'az', 'arkansas': 'ar', 'california': 'ca',
    'colorado': 'co', 'connecticut': 'ct', 'delaware': 'de', 'district of columbia': 'dc',
    'florida': 'fl', 'georgia': 'ga', 'hawaii': 'hi', 'idaho': 'id', 'illinois': 'il',
    'indiana': 'in', 'iowa': 'ia', 'kansas': 'ks', 'kentucky': 'ky', 'louisiana': 'la',
    'maine': 'me', 'maryland': 'md', 'massachusetts': 'ma', 'michigan': 'mi', 'minnesota': 'mn',
    'mississippi': 'ms', 'missouri': 'mo', 'montana': 'mt', 'nebraska': 'ne', 'nevada': 'nv',
    'new hampshire': 'nh', 'new jersey': 'nj', 'new mexico': 'nm', 'new york': 'ny',
    'north carolina': 'nc', 'north dakota': 'nd', 'ohio': 'oh', 'oklahoma': 'ok', 'oregon': 'or',
    'pennsylvania': 'pa', 'rhode island': 'ri', 'south carolina': 'sc', 'south dakota': 'sd',
    'tennessee': 'tn', 'texas': 'tx', 'utah': 'ut', 'vermont': 'vt', 'virginia': 'va',
    'washington': 'wa', 'west virginia': 'wv', 'wisconsin': 'wi', 'wyoming': 'wy',
}


# --------------------------------------------------------------------------- learning
def learn_dictionaries(s1, queries, labels, sample=400_000, seed=0):
    """Learn transliteration maps from training ground truth only.

    tokens:     native-script name token -> Latin token (positional alignment of
                equal-length name pairs; >99.99% of Indic pairs are equal length)
    components: address component (native script, state codes, ...) -> canonical
                component of the owner, from the last address components
    """
    pairs = (labels.join(queries.select('entity_id', 'business_name', 'business_address'), on='entity_id')
             .join(s1.select(pl.col('entity_id').alias('owner'), pl.col('business_name').alias('owner_name'),
                             pl.col('business_address').alias('owner_address')), on='owner'))
    native = pairs.filter(pl.col('business_name').str.contains(NATIVE))
    votes = defaultdict(Counter)
    for name, owner in native.select('business_name', 'owner_name').iter_rows():
        a, b = name.split(), owner.split()
        if len(a) != len(b):
            continue
        for x, y in zip(a, b):
            if re.search(NATIVE, x):
                votes[x][y.lower().strip('.,()')] += 1
    tokens = {}
    for tok, c in votes.items():
        word, n = c.most_common(1)[0]
        if n >= 2 and n / sum(c.values()) >= 0.6:
            tokens[tok] = word
    comp_votes = defaultdict(Counter)
    sampled = pairs.sample(min(sample, pairs.height), seed=seed)
    for addr, owner in sampled.select('business_address', 'owner_address').iter_rows():
        a = [x.strip() for x in addr.split(',') if x.strip()]
        b = [x.strip() for x in owner.split(',') if x.strip()]
        if a and b:
            comp_votes[_component_key(a[-1])][_component_key(b[-1])] += 1
    # A source must be a query-side spelling (state code, native script, full US state
    # name), never a locality that S1 itself uses (a city mapped to its state loses info).
    s1_usage = Counter()
    for addr in s1['business_address'].sample(min(sample, s1.height), seed=seed).to_list():
        s1_usage.update(_component_key(x) for x in addr.split(',') if x.strip())
    scale = s1.height / max(1, min(sample, s1.height))
    components = {}
    for comp, c in comp_votes.items():
        target, n = c.most_common(1)[0]
        total = sum(c.values())
        if (comp and target and comp != target and n >= 20 and n / total >= 0.6
                and s1_usage[comp] * scale < 0.05 * total * pairs.height / max(1, sampled.height)):
            components[comp] = target
    return {'tokens': tokens, 'components': components}


def _component_key(s):
    return ' '.join(re.findall(r'[a-z0-9]+', anyascii(s).lower())) if not re.search(NATIVE, s) else s.strip()


def save_dictionaries(path, d):
    Path(path).write_text(json.dumps(d, ensure_ascii=False), encoding='utf-8')


def load_dictionaries(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


# --------------------------------------------------------------------------- transliteration
def _translit_name(s, tokens):
    out = []
    for t in s.split():
        if re.search(NATIVE, t):
            out.append(tokens.get(t) or anyascii(t))
        else:
            out.append(anyascii(t))
    return ' '.join(out)


def _translit_address(s, tokens, components):
    parts = []
    for comp in s.split(','):
        c = comp.strip()
        if re.search(NATIVE, c):
            c = components.get(c) or ' '.join(tokens.get(t) or anyascii(t) for t in c.split())
        else:
            c = anyascii(c)
        parts.append(c)
    return ','.join(parts)


def _translit_column(col, fn):
    """Apply fn once per distinct non-ASCII value; ASCII values pass through."""
    values = col.filter(col.str.contains(r'[^\x00-\x7F]')).unique()
    mapping = {v: fn(v) for v in values.to_list()}
    if not mapping:
        return col
    return col.replace(pl.Series(list(mapping.keys())), pl.Series(list(mapping.values())))


# --------------------------------------------------------------------------- names
def _name_views(name):
    """Vectorised name normalisation on an already transliterated Series."""
    df = pl.DataFrame({'raw': name.str.to_lowercase().str.strip_chars()})
    domain = r'^(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9-]*)\.(?:com|net|org|in|co\.in|co|biz|info|io|fr|us)$'
    tld = r'\.(?:com|net|org|co\.in|in|biz|info|io|fr|us)\b'
    df = df.with_columns(
        (pl.col('raw').str.contains(domain) | pl.col('raw').str.contains(tld)).alias('f_domain'),
        pl.col('raw').str.contains(r'^#\w+$').alias('f_hash'),
    ).with_columns(
        pl.when(pl.col('raw').str.contains(domain)).then(pl.col('raw').str.extract(domain, 1))
        .otherwise(pl.col('raw').str.replace_all(tld, ' ').str.replace_all(r'^(?:https?://)?www\.', '')).alias('s'))
    s = (pl.col('s').str.replace_all(r'[&+]', ' and ').str.replace_all(r"['’`]", '')
         .str.replace_all(r'[^a-z0-9]+', ' ').str.strip_chars())
    df = df.with_columns(s.alias('s'))
    # Phone numbers appended to names ("back alley tattoo 4512252403") are noise.
    df = df.with_columns(pl.col('s').str.replace_all(r'\b\d{7,}\b', ' ').str.replace_all(r'\s+', ' ').str.strip_chars())
    for a, b in SPACED:
        df = df.with_columns(pl.col('s').str.replace_all(r'\b' + a + r'\b', b))
    # Leet substitutions only inside mixed letter/digit tokens (0->o, 1->l, 5->s ...).
    leet = (pl.col('s').str.split(' ').list.eval(
        pl.when(pl.element().str.contains(r'[a-z]') & pl.element().str.contains(r'\d'))
        .then(pl.element().str.replace_many(['0', '1', '3', '4', '5', '7', '8'], ['o', 'l', 'e', 'a', 's', 't', 'b']))
        .otherwise(pl.element())).list.join(' '))
    df = df.with_columns(leet.alias('s'))
    parts = pl.col('s').str.extract(r'^(.*?)' + ALIAS_PHRASE + r'(.*)$', 1)
    after = pl.col('s').str.extract(r'^.*?' + ALIAS_PHRASE + r'(.*)$', 1).str.strip_chars()
    df = df.with_columns(
        (after.is_not_null() & (after != '')).alias('f_dba'),
        parts.str.strip_chars().fill_null('').alias('alt'),
        pl.when(after.is_not_null() & (after != '')).then(after).otherwise(pl.col('s')).alias('nm'))
    tok = pl.col('nm').str.split(' ').list.eval(pl.element().replace(LEGAL))
    df = df.with_columns(tok.alias('tok'))
    legal_set = pl.lit(LEGAL_FORMS)
    stop = pl.lit(sorted(NAME_STOP))
    df = df.with_columns(
        pl.col('tok').list.join(' ').alias('nm'),
        pl.col('tok').list.set_intersection(legal_set).list.sort().list.join(' ').alias('legal'),
        pl.col('tok').list.eval(pl.element().filter(~pl.element().is_in(LEGAL_FORMS) & ~pl.element().is_in(sorted(NAME_STOP)))
                                ).list.join(' ').alias('core'))
    df = df.with_columns(pl.when(pl.col('core') == '').then(pl.col('nm')).otherwise(pl.col('core')).alias('core'))
    df = df.with_columns(pl.col('core').str.replace_all(' ', '').alias('sq'))
    return df.select('nm', 'core', 'sq', 'alt', 'legal', 'f_domain', 'f_hash', 'f_dba')


# --------------------------------------------------------------------------- addresses
def _normalise_component(expr, components):
    """Whole-component canonical map first (states, native regions), else token map."""
    plain = (expr.str.to_lowercase().str.replace_all(r"['’`]", '').str.replace_all(r'(\d)(?:st|nd|rd|th)\b', '${1}')
             .str.extract_all(r'[a-z0-9]+').list.join(' '))
    mapped = plain.replace(components).replace(US_STATES)
    toks = (plain.str.extract_all(r'\d+|[a-z]+')
            .list.eval(pl.when(pl.element().str.contains(r'^\d+$'))
                       .then(pl.element().str.strip_chars_start('0').replace('', '0'))
                       .otherwise(pl.element().replace(STREET))).list.join(' '))
    return pl.when(mapped != plain).then(mapped).otherwise(toks)


def component_frequencies(address, components):
    """Count normalised address components (no digits) in one split's catalogue."""
    comps = (pl.DataFrame({'a': address}).select(pl.col('a').str.split(',').alias('c')).explode('c')
             .select(_normalise_component(pl.col('c'), components).alias('c')).filter(pl.col('c') != '')
             .filter(~pl.col('c').str.contains(r'\d')))
    return comps.group_by('c').len()


def _address_views(address, components, freq, threshold):
    """components: learned canonical map; freq: component counts (locality detector)."""
    n = len(address)
    comp = (pl.DataFrame({'a': address}).with_row_index('row')
            .select('row', pl.col('a').str.split(',').alias('c')).explode('c')
            .with_columns(_normalise_component(pl.col('c'), components).alias('c'))
            .filter(pl.col('c') != '')
            .filter(~pl.col('c').is_in(['null', 'n a', 'na', 'none', 'nil'])))
    comp = comp.join(freq.rename({'len': 'freq'}), on='c', how='left').with_columns(pl.col('freq').fill_null(0))
    comp = comp.with_columns(((pl.col('freq') >= threshold) & ~pl.col('c').str.contains(r'\d')).alias('is_loc'))
    agg = comp.group_by('row').agg(
        pl.col('c').filter(~pl.col('is_loc')).str.join(' ').alias('street'),
        pl.col('c').filter(pl.col('is_loc')).str.join(' ').alias('loc'),
        pl.col('c').str.join(' ').alias('ad'))
    out = pl.DataFrame({'row': pl.arange(0, n, eager=True).cast(pl.UInt32)}).join(agg, on='row', how='left')
    out = out.with_columns(pl.col('street', 'loc', 'ad').fill_null(''))
    drop = sorted(ADDR_NOISE)
    out = out.with_columns(
        pl.col('street').str.split(' ').list.eval(pl.element().filter(~pl.element().is_in(drop) & (pl.element() != '')))
        .list.join(' ').alias('street'),
        pl.col('ad').str.extract_all(r'\b\d+\b').list.unique(maintain_order=True).list.join(' ').alias('nums'),
        pl.col('street').str.extract(r'\b(\d+)\b', 1).fill_null('').alias('hn'))
    return out.select('ad', 'street', 'loc', 'nums', 'hn')


# --------------------------------------------------------------------------- public API
def normalise(df, dictionaries, freq, loc_threshold=40):
    """df: entity_id, business_name, business_address, country -> canonical views."""
    tokens, components = dictionaries['tokens'], dictionaries['components']
    name = _translit_column(df['business_name'], lambda s: _translit_name(s, tokens))
    addr = _translit_column(df['business_address'], lambda s: _translit_address(s, tokens, components))
    nv = _name_views(name)
    av = _address_views(addr, components, freq, loc_threshold)
    base = df.select('entity_id', pl.col('entity_id').str.slice(0, 2).alias('src'),
                     pl.col('country').str.to_lowercase().str.strip_chars().alias('country'),
                     df['business_name'].str.contains(NATIVE).alias('f_native'),
                     (df['business_address'].str.strip_chars() == '').alias('f_noaddr'))
    return pl.concat([base, nv, av], how='horizontal')


def _components_frame(df, tokens, components):
    """One row per (record, address component): country, plain component key, position flags."""
    addr = _translit_column(df['business_address'], lambda s: _translit_address(s, tokens, components))
    c = (pl.DataFrame({'country': df['country'].str.to_lowercase().str.strip_chars(), 'a': addr}).with_row_index('row')
         .with_columns(pl.col('a').str.split(',')).explode('a')
         .with_columns(pl.col('a').str.to_lowercase().str.extract_all(r'[a-z0-9]+').list.join(' ').alias('c'))
         .filter((pl.col('c') != '') & ~pl.col('c').str.contains(r'\d')))
    c = c.with_columns(pl.col('c').replace(components).replace(US_STATES))
    return c.with_columns((pl.int_range(pl.len()).over('row') == pl.len().over('row') - 1).alias('is_last')).drop('a')


def learn_region_map(s1, q, dictionaries, min_votes=30):
    """Unsupervised region equivalences from one split's own records (no labels).

    S1 writes regions ("Nouvelle-Aquitaine") where S2/S3 often write a department
    ("Gironde"). A query-side last component that S1 never uses as a region is mapped
    to the region that S1 assigns to the cities it co-occurs with. Learned per split,
    so it also covers France, which has no training labels."""
    tokens, components = dictionaries['tokens'], dictionaries['components']
    sc = _components_frame(s1, tokens, components)
    regions = sc.filter(pl.col('is_last')).group_by('country', 'c').len().filter(pl.col('len') >= 50).select('country', 'c')
    city = (sc.filter(~pl.col('is_last')).join(sc.filter(pl.col('is_last')).select('row', pl.col('c').alias('r')), on='row')
            .group_by('country', 'c', 'r').len())
    city = (city.with_columns((pl.col('len') / pl.col('len').sum().over('country', 'c')).alias('share'),
                              pl.col('len').sum().over('country', 'c').alias('n'))
            .filter((pl.col('share') >= 0.9) & (pl.col('n') >= 20)).select('country', 'c', 'r'))
    seen = sc.select('country', 'c').unique()  # never remap anything S1 itself writes
    maps = {}
    for lo in range(0, q.height, 2_000_000):
        qc = _components_frame(q.slice(lo, 2_000_000), tokens, components)
        cand = (qc.filter(pl.col('is_last')).join(regions, on=['country', 'c'], how='anti')
                .join(seen, on=['country', 'c'], how='anti').select('row', 'country', pl.col('c').alias('x')))
        votes = (cand.join(qc.filter(~pl.col('is_last')).select('row', 'c'), on='row')
                 .join(city, on=['country', 'c']).group_by('country', 'x', 'r').len())
        maps[lo] = votes
    if not maps:
        return {}
    v = pl.concat(list(maps.values())).group_by('country', 'x', 'r').agg(pl.col('len').sum())
    v = v.with_columns((pl.col('len') / pl.col('len').sum().over('country', 'x')).alias('share'),
                       pl.col('len').sum().over('country', 'x').alias('n'))
    v = v.filter((pl.col('share') >= 0.8) & (pl.col('n') >= min_votes) & (pl.col('x') != pl.col('r')))
    return dict(zip(v['x'].to_list(), v['r'].to_list()))


def read_tsv(path):
    return (pl.read_csv(path, separator='\t', infer_schema=False, quote_char=None,
                        missing_utf8_is_empty_string=True).fill_null(''))


def prepare_all(data, out, chunk=1_000_000, log=print):
    """Raw challenge TSVs -> normalised parquet (the files shipped in prepared/).

    Writes: dictionaries.json, train_s1.parquet, train_q.parquet (with 'owner', '' if
    unmatched), test_s1.parquet, test_q.parquet, and PREP_VERSION.
    Dictionaries use training labels only; address locality frequencies are computed
    per split from that split's own records (no labels involved)."""
    data, out = Path(data), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tr1 = read_tsv(data / 'train/train_source1.tsv')
    trq = pl.concat([read_tsv(data / 'train/train_source2.tsv'), read_tsv(data / 'train/train_source3.tsv')])
    gt = read_tsv(data / 'train/train_ground_truth.tsv')
    labels = (gt.with_columns(pl.col('matched_entity_ids').str.split(',')).explode('matched_entity_ids')
              .filter(pl.col('matched_entity_ids') != '')
              .select(pl.col('matched_entity_ids').alias('entity_id'), pl.col('source1_entity_id').alias('owner')))
    if labels['entity_id'].n_unique() != labels.height:
        raise ValueError('ground truth assigns a query to several S1 entities')
    d = learn_dictionaries(tr1, trq, labels)
    save_dictionaries(out / 'dictionaries.json', d)
    log(f'dictionaries: {len(d["tokens"])} tokens, {len(d["components"])} components')
    trq = trq.join(labels, on='entity_id', how='left').with_columns(pl.col('owner').fill_null(''))
    for split, s1, q in [('train', tr1, trq), ('test', None, None)]:
        if split == 'test':
            del tr1, trq
            s1 = read_tsv(data / 'test/test_source1.tsv')
            q = pl.concat([read_tsv(data / 'test/test_source2.tsv'), read_tsv(data / 'test/test_source3.tsv')])
        regions = learn_region_map(s1, q, d)
        ds = {'tokens': d['tokens'], 'components': {**d['components'], **regions}}
        d[f'regions_{split}'] = regions
        save_dictionaries(out / 'dictionaries.json', d)
        log(f'{split}: {len(regions)} unsupervised region equivalences, e.g. {list(regions.items())[:8]}')
        freq = address_frequencies([s1, q], ds)
        normalise(s1, ds, freq).write_parquet(out / f'{split}_s1.parquet')
        parts = []
        for lo in range(0, q.height, chunk):
            c = q.slice(lo, chunk)
            v = normalise(c.select('entity_id', 'business_name', 'business_address', 'country'), ds, freq)
            if 'owner' in c.columns:
                v = v.with_columns(c['owner'])
            parts.append(v)
        pl.concat(parts).write_parquet(out / f'{split}_q.parquet')
        log(f'prepared {split}: {s1.height:,} S1, {q.height:,} queries')
    (out / 'PREP_VERSION').write_text(PREP_VERSION)


def address_frequencies(frames, dictionaries, chunk=1_000_000):
    """Component frequency table over all records of one split (S1+S2+S3)."""
    tokens, components = dictionaries['tokens'], dictionaries['components']
    total = None
    for f in frames:
        for lo in range(0, f.height, chunk):
            col = f['business_address'].slice(lo, chunk)
            addr = _translit_column(col, lambda s: _translit_address(s, tokens, components))
            part = component_frequencies(addr, components)
            total = part if total is None else pl.concat([total, part]).group_by('c').agg(pl.col('len').sum())
    return total
