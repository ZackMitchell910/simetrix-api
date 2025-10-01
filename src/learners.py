import numpy as np
from dataclasses import dataclass

def _sigmoid(z: float) -> float:
    # numerically stable logistic
    z = float(np.clip(z, -60.0, 60.0))
    if z >= 0:
        ez = np.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = np.exp(z)
        return ez / (1.0 + ez)

@dataclass
class OnlineLinear:
    lr: float = 0.05
    l2: float = 1e-4
    w: np.ndarray | None = None  # shape: (1 + n_features,)

    def init(self, n_features: int):
        """Ensure weights exist with correct length (bias + features)."""
        n = int(n_features) + 1
        if self.w is None:
            self.w = np.zeros(n, dtype=float)
        else:
            # pad or truncate to match requested size
            if self.w.size < n:
                self.w = np.pad(self.w.astype(float), (0, n - self.w.size))
            elif self.w.size > n:
                self.w = self.w[:n].astype(float)

    def _xb(self, x: np.ndarray) -> np.ndarray:
        """Return [1, x...] aligned to current weight length."""
        x = np.asarray(x, dtype=float).ravel()
        # if uninitialized, assume x length
        if self.w is None:
            self.init(x.size)
        # align to len(w)
        xb = np.concatenate([[1.0], x])
        if xb.size < self.w.size:
            xb = np.pad(xb, (0, self.w.size - xb.size))
        elif xb.size > self.w.size:
            xb = xb[: self.w.size]
        # clean NaN/Inf to zeros
        if not np.all(np.isfinite(xb)):
            xb = np.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0)
        return xb

    def proba(self, x: np.ndarray) -> float:
        xb = self._xb(x)
        z = float(np.dot(self.w, xb))
        return _sigmoid(z)

    def update(self, x: np.ndarray, y: float):
        """One SGD step on logistic loss with L2."""
        xb = self._xb(x)
        y = float(y)
        p = _sigmoid(float(np.dot(self.w, xb)))
        # gradient: (p - y)*xb + l2*w
        g = (p - y) * xb + self.l2 * self.w
        self.w -= self.lr * g

@dataclass
class ExpWeights:
    eta: float = 2.0
    w: np.ndarray | None = None  # shape: (n_models,)

    def init(self, n_models: int):
        n = int(n_models)
        if n <= 0:
            self.w = np.ones(1, dtype=float)
            return
        if self.w is None or self.w.size != n:
            self.w = np.ones(n, dtype=float) / n
        else:
            # re-normalize just in case
            s = float(self.w.sum())
            self.w = (self.w / s) if s > 0 else (np.ones(n, dtype=float) / n)

    def update(self, losses: np.ndarray):
        """Exponentially weight models by (negative) losses, normalized safely."""
        losses = np.asarray(losses, dtype=float).ravel()
        if self.w is None:
            self.init(len(losses))
        # align lengths (pad/truncate)
        n = self.w.size
        if losses.size < n:
            losses = np.pad(losses, (0, n - losses.size), constant_values=np.mean(losses) if losses.size else 0.0)
        elif losses.size > n:
            losses = losses[:n]
        # clean NaN/Inf
        if not np.all(np.isfinite(losses)):
            m = float(np.nanmean(losses)) if np.isfinite(np.nanmean(losses)) else 0.0
            losses = np.nan_to_num(losses, nan=m, posinf=m, neginf=m)
        # update + normalize
        self.w *= np.exp(-self.eta * losses)
        s = float(self.w.sum())
        if s <= 0 or not np.isfinite(s):
            self.w = np.ones_like(self.w) / n
        else:
            self.w /= s
