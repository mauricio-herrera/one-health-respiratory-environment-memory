# Methods freeze

M0 = six seasonal harmonics + six day-of-week nuisance dummies + fifteen unit
nuisance dummies (27 design columns).

M1 adds seven contemporaneous environmental/vulnerability predictors (34 total).

M2 adds twenty environmental-memory predictors (54 total).

M3 adds fifteen lagged health-memory predictors (69 total).

Prospective full-history M0–M2 repair fit:
- PoissonRegressor
- alpha = 1e-5
- tol = 1e-10
- max_iter = 800
- fit_intercept = True
- solver = lbfgs
- target = urgent_respiratory_total / population
- sample_weight = population
- predicted count = predicted rate × population

External uncertainty:
- paired moving-block bootstrap
- block length = 7 calendar days
- B = 5000
- seed = 20260822
- all 11 regions preserved together within resampled day
- two-sided 95% percentile CIs
- RMSE recomputed as sqrt(mean(resampled MSE))

No post-hoc tuning is authorized.
