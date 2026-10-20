# Randomization inference

A randomization test compares the observed treatment contrast with contrasts
under new assignments from the same design. Redrawing all treatment labels
tests the **global sharp null**: no participant's outcome would change under
any treatment. A statistic comparing two arms does not make this a test that
allows arbitrary treatment effects in the other arms.

This complete example simulates an experiment, then tests its outcome:

```python
import numpy as np

from randaba import designs, inference

rng = np.random.default_rng(42)
x = rng.normal(size=(800, 5))
assignment = designs.randomized_aba(x, k=20, rng=rng)

# Simulated observed outcomes; use your experiment's outcomes here.
y = x.sum(axis=1) + 0.6 * (assignment == 3) + rng.normal(size=len(x))
estimate = inference.contrast(y, assignment, 3, 0)
reference = np.array([designs.randomized_aba(x, k=20, rng=rng) for _ in range(2000)])
pvalue = inference.ri_pvalue(estimate, inference.contrast(y, reference, 3, 0))
print(estimate, pvalue)
```

Use exactly the same input features, arm count, and jitter setting as in the
assignment actually run. Hold the observed outcomes fixed when computing
reference contrasts. If you balanced `x` and `x**2`, redraw on those same
combined features.

The two-sided Monte Carlo p-value is `(1 + exceedances) / (1 + draws)`;
ties count as exceedances. The added observed draw prevents zero p-values
and gives a valid finite-sample test under the global sharp null. Discrete
p-values can make it conservative. More reference draws give finer p-value
resolution.

Unbiased estimation and test validity follow from the design, not from the
simulation results. Uniform label shuffling makes each participant equally
likely to receive each treatment. With equal group sizes, the difference in
means therefore estimates the average treatment contrast without bias
under the usual fixed-potential-outcomes setup.
