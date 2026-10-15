"""How random is each design at small N? Distinct splits, inclusion, balance, time."""

import time

import numpy as np
from registry import DESIGNS

R = 2000
for n, p in [(20, 1), (20, 5), (50, 5), (100, 20)]:
    x = np.random.default_rng(100 + n + p).normal(size=(n, p))
    sinv = np.linalg.inv(np.atleast_2d(np.cov(x, rowvar=False)))
    possible = 2 ** (n - 1) if n <= 50 else "huge"
    print(f"\nN={n} p={p}   ({R} draws; distinct partitions possible: {possible})")
    for name, draw in DESIGNS.items():
        rng = np.random.default_rng(1)
        t = time.perf_counter()
        zs = np.array([draw(x, rng) for _ in range(R)])
        el = (time.perf_counter() - t) / R
        # a split and its label swap are the same partition
        canon = {tuple(z if z[0] == 0 else 1 - z) for z in zs}
        incl = zs.mean(0)
        d = (zs @ x) / (n / 2) - ((1 - zs) @ x) / (n / 2)
        m = np.einsum("ij,jk,ik->i", d, sinv, d) * n / 4
        print(
            f"  {name:10s} distinct partitions {len(canon):5d}  "
            f"inclusion {incl.min():.2f}-{incl.max():.2f}  "
            f"mean Mahalanobis {m.mean():8.4f}  {el * 1e3:7.2f} ms/draw",
            flush=True,
        )
