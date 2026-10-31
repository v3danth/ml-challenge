"""Dense entity encoder (multilingual-E5-small, MIT) fine-tuned contrastively on
(query record, owner S1 record) pairs, plus exact per-country GPU k-NN.

The encoder reads the *normalised* views (transliterated, canonical), so every
source looks alike. Exact brute force on the GPU is cheap at this scale
(e.g. 3.8M US queries x 663k US S1 x 384 dims is seconds on an RTX 6000), so no
approximate index and no recall lost to one.
"""
import gc
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def texts(v):
    return v.select(pl.concat_str([pl.col('nm'), pl.lit(' ; '), pl.col('street'), pl.lit(' ; '), pl.col('loc')])
                    .str.slice(0, 300).alias('t'))['t'].to_list()


class Encoder:
    def __init__(self, path, max_len=64, pooling='mean', prefix='', device=None):
        """pooling: 'mean' (E5) or 'cls' (LaBSE). prefix: prepended to every input, e.g. the
        "query: " E5 was pre-trained with; the task is symmetric, so both sides get it."""
        if pooling not in ('mean', 'cls'):
            raise ValueError(f'unknown pooling {pooling!r}')
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path).to(self.device)
        self.max_len = max_len
        self.pooling = pooling
        self.prefix = prefix
        self.amp = self.device == 'cuda'

    def save(self, path):
        self.model.save_pretrained(path, safe_serialization=True)
        self.tok.save_pretrained(path)

    def _embed(self, batch):
        if self.prefix:
            batch = [self.prefix + s for s in batch]
        t = self.tok(batch, padding=True, truncation=True, max_length=self.max_len, return_tensors='pt').to(self.device)
        with torch.autocast('cuda', dtype=torch.float16, enabled=self.amp):
            h = self.model(**t).last_hidden_state
        if self.pooling == 'cls':
            return F.normalize(h[:, 0].float(), dim=-1)
        m = t['attention_mask'].unsqueeze(-1).to(h.dtype)
        return F.normalize((h * m).sum(1).float() / m.sum(1).float().clamp(min=1), dim=-1)

    @torch.no_grad()
    def encode(self, strings, batch=1024):
        self.model.eval()
        order = np.argsort([len(s) for s in strings], kind='stable')  # length bucketing
        out = np.empty((len(strings), self.model.config.hidden_size), np.float16)
        lo = 0
        while lo < len(strings):
            idx = order[lo:lo + batch]
            try:
                out[idx] = self._embed([strings[i] for i in idx]).cpu().numpy().astype(np.float16)
                lo += len(idx)
            except torch.cuda.OutOfMemoryError:
                gc.collect(); torch.cuda.empty_cache()
                if batch <= 8:
                    raise
                batch //= 2
        return out

    def train(self, anchors, positives, hard, epochs=1, batch=512, lr=5e-5, temperature=0.05, seed=0, log=print):
        """Symmetric InfoNCE. anchors[i] ~ positives[i]; hard[i] is a lexically similar,
        wrong S1 record for anchor i (sibling decoys are the main precision risk)."""
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        n = len(anchors)
        rng = np.random.default_rng(seed)
        scaler = torch.amp.GradScaler('cuda', enabled=self.amp)
        step, t0 = 0, time.time()
        total = max(1, epochs * (n // batch))
        for _ in range(epochs):
            perm = rng.permutation(n)
            lo = 0
            while lo + batch <= n:
                idx = perm[lo:lo + batch]
                frac = step / total
                for g in opt.param_groups:
                    g['lr'] = lr * min(1.0, (step + 1) / (0.05 * total)) * max(0.02, 1 - frac)
                try:
                    self.model.train()
                    a = self._embed([anchors[i] for i in idx])
                    p = self._embed([positives[i] for i in idx] + [hard[i] for i in idx])
                    logits = a @ p.T / temperature
                    target = torch.arange(len(idx), device=self.device)
                    loss = F.cross_entropy(logits, target) + F.cross_entropy(logits[:, :len(idx)].T, target)
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                except torch.cuda.OutOfMemoryError:
                    opt.zero_grad(set_to_none=True)
                    a = p = logits = loss = None
                    gc.collect(); torch.cuda.empty_cache()
                    if batch <= 32:
                        raise
                    batch //= 2
                    total = max(1, epochs * (n // batch))
                    log(f'encoder OOM: batch -> {batch}')
                    continue
                lo += batch
                step += 1
                if step % 200 == 0:
                    log(f'encoder step {step}/{total} loss {loss.item():.4f} {time.time() - t0:.0f}s')
        self.model.eval()


def knn_candidates(q, qemb, r, remb, k, device=None, q_chunk=1024):
    """Exact inner-product top-k within each country.
    q, r: frames with local row id 'i' and 'country'. Returns qi, ri, cos, drank."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16 if device == 'cuda' else torch.float32
    parts = []
    for country in q['country'].unique().to_list():
        qi = q.filter(pl.col('country') == country)['i'].to_numpy()
        ri = r.filter(pl.col('country') == country)['i'].to_numpy()
        if len(qi) == 0 or len(ri) == 0:
            continue
        kk = min(k, len(ri))
        R = torch.from_numpy(remb[ri]).to(device=device, dtype=dtype)
        for lo in range(0, len(qi), q_chunk):
            sub = qi[lo:lo + q_chunk]
            Q = torch.from_numpy(qemb[sub]).to(device=device, dtype=dtype)
            s, ix = (Q @ R.T).float().topk(kk, dim=1)
            parts.append(pl.DataFrame({
                'qi': np.repeat(sub, kk).astype(np.uint32),
                'ri': ri[ix.cpu().numpy().ravel()].astype(np.uint32),
                'cos': s.cpu().numpy().ravel().astype(np.float32),
                'drank': np.tile(np.arange(kk, dtype=np.float32), len(sub))}))
        del R
        if device == 'cuda':
            torch.cuda.empty_cache()
    if not parts:
        return pl.DataFrame(schema={'qi': pl.UInt32, 'ri': pl.UInt32, 'cos': pl.Float32, 'drank': pl.Float32})
    return pl.concat(parts)
