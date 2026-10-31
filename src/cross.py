"""Cross-encoder: reads (query record, candidate S1 record) jointly and outputs a match logit.

Initialised from multilingual-E5-small (MIT) with a linear head on mean-pooled
states, fine-tuned with binary cross-entropy on the pruned candidate pairs, which
are exactly the hard cases left after blocking (siblings, decoys, typos). Used as
one more feature for the gradient-boosted matcher, always out-of-fold on train.
"""
import gc
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def pair_texts(views):
    return views.select(pl.concat_str([pl.col('nm'), pl.lit(' | '), pl.col('street'), pl.lit(' | '), pl.col('loc')])
                        .str.slice(0, 260).alias('t'))['t'].to_list()


class CrossEncoder(torch.nn.Module):
    def __init__(self, path, max_len=128, device=None):
        super().__init__()
        self.device_name = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tok = AutoTokenizer.from_pretrained(path)
        self.enc = AutoModel.from_pretrained(path)
        self.head = torch.nn.Linear(self.enc.config.hidden_size, 1)
        head = Path(path) / 'ce_head.pt'
        if head.exists():
            self.head.load_state_dict(torch.load(head, map_location='cpu'))
        self.max_len = max_len
        self.to(self.device_name)
        self.amp = self.device_name == 'cuda'

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.enc.save_pretrained(path, safe_serialization=True)
        self.tok.save_pretrained(path)
        torch.save(self.head.state_dict(), path / 'ce_head.pt')

    def logits(self, a, b):
        t = self.tok(a, b, padding=True, truncation='longest_first', max_length=self.max_len,
                     return_tensors='pt').to(self.device_name)
        with torch.autocast('cuda', dtype=torch.float16, enabled=self.amp):
            h = self.enc(**t).last_hidden_state
        m = t['attention_mask'].unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1).float() / m.sum(1).float().clamp(min=1)
        return self.head(pooled).squeeze(-1)

    def fit(self, a, b, y, epochs=1, batch=256, lr=3e-5, seed=0, log=print):
        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.01)
        scaler = torch.amp.GradScaler('cuda', enabled=self.amp)
        n = len(y)
        rng = np.random.default_rng(seed)
        total = max(1, epochs * (n // batch))
        step, t0 = 0, time.time()
        yt = torch.tensor(y, dtype=torch.float32)
        for _ in range(epochs):
            perm = rng.permutation(n)
            lo = 0
            while lo + batch <= n:
                idx = perm[lo:lo + batch]
                for g in opt.param_groups:
                    g['lr'] = lr * min(1.0, (step + 1) / (0.05 * total)) * max(0.02, 1 - step / total)
                try:
                    self.train()
                    z = self.logits([a[i] for i in idx], [b[i] for i in idx])
                    loss = F.binary_cross_entropy_with_logits(z, yt[idx].to(self.device_name))
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                except torch.cuda.OutOfMemoryError:
                    opt.zero_grad(set_to_none=True)
                    z = loss = None
                    gc.collect(); torch.cuda.empty_cache()
                    if batch <= 16:
                        raise
                    batch //= 2
                    total = max(1, epochs * (n // batch))
                    log(f'cross-encoder OOM: batch -> {batch}')
                    continue
                lo += batch
                step += 1
                if step % 500 == 0:
                    log(f'cross-encoder step {step}/{total} loss {loss.item():.4f} {time.time() - t0:.0f}s')
        self.eval()

    @torch.no_grad()
    def predict(self, a, b, batch=1024):
        self.eval()
        order = np.argsort([len(x) + len(y) for x, y in zip(a, b)], kind='stable')
        out = np.empty(len(a), np.float32)
        lo = 0
        while lo < len(a):
            idx = order[lo:lo + batch]
            try:
                out[idx] = self.logits([a[i] for i in idx], [b[i] for i in idx]).float().cpu().numpy()
                lo += len(idx)
            except torch.cuda.OutOfMemoryError:
                gc.collect(); torch.cuda.empty_cache()
                if batch <= 8:
                    raise
                batch //= 2
        return out
