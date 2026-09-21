"""Reproduce the dimension-11 Brownian-bridge critical values for Table 10."""
from pathlib import Path
import json
import numpy as np


def main():
    reps, grid, dim, seed, batch = 50000, 20000, 11, 20260916, 20
    rng = np.random.default_rng(seed)
    maxima = np.empty(reps)
    time = np.arange(1, grid + 1, dtype=float) / grid
    for start in range(0, reps, batch):
        size = min(batch, reps - start)
        squared = np.zeros((size, grid))
        for _ in range(dim):
            path = rng.standard_normal((size, grid)).cumsum(axis=1)
            path -= path[:, -1, None] * time
            squared += path * path / grid
        maxima[start:start + size] = squared.max(axis=1)
        if (start + size) % 5000 == 0:
            print(f'Calibrated {start + size}/{reps} bridges', flush=True)
    result = dict(dimension=dim, replications=reps, grid=grid, seed=seed,
                  critical_values={str(a): float(np.quantile(maxima, 1-a))
                                   for a in [0.1, 0.05, 0.025, 0.01]})
    out = Path(__file__).parent / 'Results'
    out.mkdir(exist_ok=True)
    (out / 'critical_values_d11.json').write_text(json.dumps(result, indent=2))
    np.save(out / 'critical_values_d11_maxima.npy', maxima)
    print(result, flush=True)


if __name__ == '__main__':
    main()

