"""Write a small, self-consistent slice of the challenge data for a quick end-to-end check.

python tools/make_smoke.py --data <dataset> --out smoke_data --percent 2
Keeps `percent`% of S1 entities with all their labelled queries, the same share of
unmatched queries, and a `percent`% slice of the test files.
"""
import argparse
from pathlib import Path

import polars as pl

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--percent', type=int, default=2)
a = ap.parse_args()
data, out = Path(a.data), Path(a.out)
for s in ('train', 'test'):
    (out / s).mkdir(parents=True, exist_ok=True)
read = lambda p: pl.read_csv(p, separator='\t', infer_schema=False, quote_char=None, missing_utf8_is_empty_string=True).fill_null('')
keep = lambda c: pl.col(c).hash(seed=5).mod(100) < a.percent
write = lambda df, p: df.select('entity_id', 'business_name', 'business_address', 'country').write_csv(p, separator='\t', quote_style='never')
s1 = read(data / 'train/train_source1.tsv').filter(keep('entity_id'))
gt_all = read(data / 'train/train_ground_truth.tsv')
labels = (gt_all.with_columns(pl.col('matched_entity_ids').str.split(',')).explode('matched_entity_ids')
          .filter(pl.col('matched_entity_ids') != '')
          .select(pl.col('matched_entity_ids').alias('entity_id'), pl.col('source1_entity_id').alias('owner')))
gt = gt_all.join(s1.select(pl.col('entity_id').alias('source1_entity_id')), on='source1_entity_id', how='semi')
write(s1, out / 'train/train_source1.tsv')
gt.write_csv(out / 'train/train_ground_truth.tsv', separator='	', quote_style='never')
for k in (2, 3):
    q = read(data / f'train/train_source{k}.tsv').join(labels, on='entity_id', how='left')
    q = q.join(s1.select(pl.col('entity_id').alias('owner'), pl.lit(True).alias('kept')), on='owner', how='left')
    write(q.filter(pl.col('kept').fill_null(False) | (pl.col('owner').is_null() & keep('entity_id'))), out / f'train/train_source{k}.tsv')
for k in (1, 2, 3):
    write(read(data / f'test/test_source{k}.tsv').filter(keep('entity_id')), out / f'test/test_source{k}.tsv')
print('smoke data written to', out)
