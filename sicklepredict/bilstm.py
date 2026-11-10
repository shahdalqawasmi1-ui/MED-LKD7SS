"""
Bidirectional LSTM with temporal attention — implemented from scratch in NumPy.

Architecture (proposal §4.1):
    24-h multichannel window  ──►  forward LSTM ─┐
                              ──►  backward LSTM ┴─► h_t (2H)  ──►  attention α_t over the 24 h
    context = Σ α_t h_t  ──►  dense(ReLU) + dropout  ──►  logit  ──►  risk score

Written in plain NumPy (no PyTorch/TensorFlow needed) so it runs on any laptop.
Gradients are derived by hand and verified numerically in `gradient_check()`.
The attention weights α_t are returned with every prediction so the dashboard can show
which hours of the window drove a risk estimate.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def _sig(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))     # numerically stable sigmoid


class BiLSTMAttention:
    def __init__(self, n_in, hidden=32, attn=32, dense=16, seed=0, dtype=np.float32):
        rng = np.random.default_rng(seed)
        self.dtype = dtype
        H = hidden

        def xavier(a, b):
            lim = np.sqrt(6 / (a + b))
            return rng.uniform(-lim, lim, (a, b)).astype(dtype)

        def lstm_bias():
            b = np.zeros(4 * H, dtype=dtype)
            b[H:2 * H] = 1.0                    # forget-gate bias = 1 (standard trick)
            return b

        self.p = {
            "Wxf": xavier(n_in, 4 * H), "Whf": xavier(H, 4 * H), "bf": lstm_bias(),
            "Wxb": xavier(n_in, 4 * H), "Whb": xavier(H, 4 * H), "bb": lstm_bias(),
            "Wa": xavier(2 * H, attn), "ba": np.zeros(attn, dtype), "va": xavier(attn, 1)[:, 0],
            "W1": xavier(2 * H, dense), "b1": np.zeros(dense, dtype),
            "w2": xavier(dense, 1), "b2": np.zeros(1, dtype),
        }
        self.H = H
        self.cfg = dict(n_in=n_in, hidden=hidden, attn=attn, dense=dense)
        self.platt = (1.0, 0.0)
        self.history = []

    # ------------------------------------------------------------------ LSTM core
    def _lstm_fwd(self, X, Wx, Wh, b):
        B, T, _ = X.shape
        H = self.H
        xs = X @ Wx + b
        h = np.zeros((B, H), self.dtype)
        c = np.zeros((B, H), self.dtype)
        hs = np.empty((B, T, H), self.dtype)
        cache = []
        for t in range(T):
            z = xs[:, t] + h @ Wh
            i = _sig(z[:, :H]); f = _sig(z[:, H:2 * H]); o = _sig(z[:, 2 * H:3 * H]); g = np.tanh(z[:, 3 * H:])
            c_prev, h_prev = c, h
            c = f * c_prev + i * g
            tc = np.tanh(c)
            h = o * tc
            hs[:, t] = h
            cache.append((i, f, o, g, c_prev, h_prev, tc))
        return hs, cache

    def _lstm_bwd(self, dhs, cache, X, Wh):
        B, T, D = X.shape
        H = self.H
        dz_all = np.empty((B, T, 4 * H), self.dtype)
        dWh = np.zeros_like(Wh)
        dh_next = np.zeros((B, H), self.dtype)
        dc_next = np.zeros((B, H), self.dtype)
        for t in range(T - 1, -1, -1):
            i, f, o, g, c_prev, h_prev, tc = cache[t]
            dh = dhs[:, t] + dh_next
            do = dh * tc
            dc = dh * o * (1 - tc * tc) + dc_next
            dz = np.concatenate([dc * g * i * (1 - i), dc * c_prev * f * (1 - f),
                                 do * o * (1 - o), dc * i * (1 - g * g)], axis=1)
            dc_next = dc * f
            dWh += h_prev.T @ dz
            dh_next = dz @ Wh.T
            dz_all[:, t] = dz
        dWx = X.reshape(-1, D).T @ dz_all.reshape(-1, 4 * H)
        db = dz_all.sum((0, 1))
        return dWx, dWh, db

    # ------------------------------------------------------------------ full model
    def forward(self, X, train=False, rng=None, dropout=0.0):
        p = self.p
        H = self.H
        hf, cf = self._lstm_fwd(X, p["Wxf"], p["Whf"], p["bf"])
        Xr = X[:, ::-1]
        hb_r, cb = self._lstm_fwd(Xr, p["Wxb"], p["Whb"], p["bb"])
        Hc = np.concatenate([hf, hb_r[:, ::-1]], axis=2)              # (B,T,2H)
        U = np.tanh(Hc @ p["Wa"] + p["ba"])                           # (B,T,A)
        e = U @ p["va"]                                               # (B,T)
        e = e - e.max(1, keepdims=True)
        a = np.exp(e); a /= a.sum(1, keepdims=True)                   # attention α
        ctx = (a[:, :, None] * Hc).sum(1)                             # (B,2H)
        if train and dropout > 0:
            m = (rng.random(ctx.shape) > dropout).astype(self.dtype) / (1 - dropout)
        else:
            m = np.ones_like(ctx)
        ctx_d = ctx * m
        z1 = ctx_d @ p["W1"] + p["b1"]
        a1 = np.maximum(z1, 0)
        logit = (a1 @ p["w2"] + p["b2"])[:, 0]
        cache = (X, Xr, cf, cb, Hc, U, a, ctx_d, m, z1, a1)
        return logit, a, cache

    def backward(self, dlogit, cache):
        p = self.p
        H = self.H
        X, Xr, cf, cb, Hc, U, a, ctx_d, m, z1, a1 = cache
        g = {}
        g["w2"] = a1.T @ dlogit[:, None]
        g["b2"] = np.array([dlogit.sum()], self.dtype)
        dz1 = (dlogit[:, None] * p["w2"].T) * (z1 > 0)
        g["W1"] = ctx_d.T @ dz1
        g["b1"] = dz1.sum(0)
        dctx = (dz1 @ p["W1"].T) * m
        da = (Hc * dctx[:, None, :]).sum(2)
        dHc = a[:, :, None] * dctx[:, None, :]
        de = a * (da - (a * da).sum(1, keepdims=True))
        g["va"] = (U * de[:, :, None]).sum((0, 1))
        dpre = (de[:, :, None] * p["va"]) * (1 - U * U)
        A = dpre.shape[2]
        g["Wa"] = Hc.reshape(-1, 2 * H).T @ dpre.reshape(-1, A)
        g["ba"] = dpre.sum((0, 1))
        dHc = dHc + dpre @ p["Wa"].T
        g["Wxf"], g["Whf"], g["bf"] = self._lstm_bwd(dHc[:, :, :H], cf, X, p["Whf"])
        g["Wxb"], g["Whb"], g["bb"] = self._lstm_bwd(np.ascontiguousarray(dHc[:, ::-1, H:]), cb, Xr, p["Whb"])
        return g

    @staticmethod
    def loss_and_grad_logit(logit, y, pos_weight):
        w = np.where(y > 0.5, pos_weight, 1.0)
        sw = w.sum()
        loss = (w * (np.logaddexp(0, logit) - y * logit)).sum() / sw
        dlogit = (w * (_sig(logit) - y) / sw).astype(logit.dtype)
        return loss, dlogit

    # ------------------------------------------------------------------ training
    def fit(self, Xtr, ytr, Xva, yva, epochs=30, batch=128, lr=2e-3, weight_decay=1e-4,
            dropout=0.2, clip=5.0, patience=6, seed=0, verbose=True, neg_ratio=None, input_noise=0.0):
        """neg_ratio: if set, every epoch uses all positive windows plus a FRESH random draw of
        neg_ratio × as many negative windows (more variety, less memorisation).
        input_noise: Gaussian noise (in personal-SD units) added to inputs during training."""
        rng = np.random.default_rng(seed)
        pos_all = np.flatnonzero(ytr == 1)
        neg_all = np.flatnonzero(ytr == 0)
        n_neg = len(neg_all) if neg_ratio is None else min(len(neg_all), int(neg_ratio * len(pos_all)))
        pos_weight = float(n_neg / max(len(pos_all), 1))
        mom = {k: np.zeros_like(v) for k, v in self.p.items()}
        vel = {k: np.zeros_like(v) for k, v in self.p.items()}
        b1, b2, eps, step = 0.9, 0.999, 1e-8, 0
        best, best_p, bad = -1.0, None, 0
        for ep in range(1, epochs + 1):
            neg = neg_all if neg_ratio is None else rng.choice(neg_all, n_neg, replace=False)
            order = rng.permutation(np.r_[pos_all, neg])
            n = len(order)
            losses = []
            for s in range(0, n, batch):
                bi = order[s:s + batch]
                xb = Xtr[bi]
                if input_noise > 0:
                    xb = xb + rng.normal(0, input_noise, xb.shape).astype(self.dtype)
                logit, _, cache = self.forward(xb, train=True, rng=rng, dropout=dropout)
                loss, dlogit = self.loss_and_grad_logit(logit, ytr[bi], pos_weight)
                g = self.backward(dlogit, cache)
                gn = np.sqrt(sum(float((v * v).sum()) for v in g.values()))
                scale = min(1.0, clip / (gn + 1e-12))
                step += 1
                lr_t = lr * np.sqrt(1 - b2 ** step) / (1 - b1 ** step)
                for k in self.p:
                    gk = g[k] * scale
                    mom[k] = b1 * mom[k] + (1 - b1) * gk
                    vel[k] = b2 * vel[k] + (1 - b2) * gk * gk
                    if k[0] in "Ww" or k == "va":        # decoupled weight decay on weights only
                        self.p[k] -= lr * weight_decay * self.p[k]
                    self.p[k] -= (lr_t * mom[k] / (np.sqrt(vel[k]) + eps)).astype(self.dtype)
                losses.append(loss)
            val_logit = self.predict_logit(Xva)
            ap = average_precision_score(yva, val_logit)
            self.history.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val_auprc": float(ap)})
            if verbose:
                print(f"  epoch {ep:2d}  loss {np.mean(losses):.4f}  val AUPRC {ap:.4f}")
            if ap > best + 1e-4:
                best, bad = ap, 0
                best_p = {k: v.copy() for k, v in self.p.items()}
            else:
                bad += 1
                if bad >= patience:
                    if verbose:
                        print(f"  early stop (best val AUPRC {best:.4f})")
                    break
        self.p = best_p
        return self

    # ------------------------------------------------------------------ inference
    def predict_logit(self, X, batch=4096, return_attention=False):
        logits, atts = [], []
        for s in range(0, len(X), batch):
            lg, a, _ = self.forward(X[s:s + batch])
            logits.append(lg)
            if return_attention:
                atts.append(a)
        out = np.concatenate(logits)
        return (out, np.concatenate(atts)) if return_attention else out

    def calibrate(self, logit_val, y_val):
        """Platt scaling on the validation set so the risk score is a probability."""
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit_val[:, None], y_val)
        self.platt = (float(lr.coef_[0, 0]), float(lr.intercept_[0]))
        return self

    def predict_proba(self, X, return_attention=False):
        r = self.predict_logit(X, return_attention=return_attention)
        lg, att = (r if return_attention else (r, None))
        a, b = self.platt
        p = _sig(a * lg + b)
        return (p, att) if return_attention else p

    # ------------------------------------------------------------------ persistence
    def save(self, path):
        np.savez_compressed(path, **self.p, _platt=np.array(self.platt),
                            _cfg=np.array([self.cfg[k] for k in ("n_in", "hidden", "attn", "dense")]))

    @classmethod
    def load(cls, path):
        z = np.load(path)
        n_in, hidden, attn, dense = [int(v) for v in z["_cfg"]]
        m = cls(n_in, hidden, attn, dense)
        for k in m.p:
            m.p[k] = z[k]
        m.platt = tuple(float(v) for v in z["_platt"])
        return m


def gradient_check(seed=0):
    """Compares analytic gradients with finite differences (float64). Returns max rel. error."""
    rng = np.random.default_rng(seed)
    m = BiLSTMAttention(5, hidden=4, attn=3, dense=3, seed=seed, dtype=np.float64)
    X = rng.normal(size=(6, 7, 5))
    y = (rng.random(6) > 0.5).astype(float)

    def L():
        lg, _, _ = m.forward(X)
        return m.loss_and_grad_logit(lg, y, 2.0)[0]

    lg, _, cache = m.forward(X)
    _, dl = m.loss_and_grad_logit(lg, y, 2.0)
    g = m.backward(dl, cache)
    worst = 0.0
    for k, v in m.p.items():
        flat = v.reshape(-1)
        for idx in rng.choice(flat.size, min(6, flat.size), replace=False):
            old = flat[idx]
            flat[idx] = old + 1e-6; lp = L()
            flat[idx] = old - 1e-6; lm = L()
            flat[idx] = old
            num = (lp - lm) / 2e-6
            ana = g[k].reshape(-1)[idx]
            rel = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
            worst = max(worst, rel)
    return worst


if __name__ == "__main__":
    print("max relative gradient error:", gradient_check())
