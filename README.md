# Irrigation RL: Cross-Regime Generalization Study

## What this project does

This project trains four deep reinforcement learning algorithms
(PPO, SAC, A2C, DQN) to schedule daily crop irrigation, then asks a
question most RL-for-agriculture work skips: **does a policy trained
under one weather pattern still work when the weather changes?**

Concretely, it:

1. Simulates a 150-day growing season with a daily soil-water balance
   model, where the agent chooses how many mm to irrigate each day
   based on soil moisture, weather, and crop stress.
2. Trains all four algorithms **only** on a "normal" synthetic weather
   distribution (5 random seeds each).
3. Evaluates every trained policy on three **weather regimes it never
   saw during training** -- drought, heatwave, excess rainfall -- to
   test out-of-distribution robustness.
4. Benchmarks RL against four non-learned baselines (fixed-rate,
   threshold-rule, random) and against a real-weather-driven
   "observed practice" comparison built from actual historical
   weather data (Dhaka, Bangladesh, Jan-May 2023).
5. Runs statistical significance tests (Mann-Whitney U) across seeds,
   rather than reporting single-run numbers as if they were exact.

## Results

### In-distribution performance (normal weather)

| Method | Yield | Water (mm) | Water productivity |
|---|---|---|---|
| DQN | 94.1 | 655 | 0.148 |
| fixed_15mm | 95.4 | 2250 | 0.042 |
| random | 94.7 | 2239 | 0.043 |
| PPO | 90.4 | 616 | 0.150 |
| SAC | 89.7 | 521 | 0.174 |
| A2C | 37.1* | 262 | 0.116* |
| threshold | 26.2 | 368 | 0.079 |

*A2C's average is depressed by a seed-level failure mode -- see below.

All four RL algorithms match brute-force baselines' yield using
roughly a quarter to a third of the water.

### Cross-regime generalization -- the headline result

| Method | Normal | Drought | Heatwave | Excess rain |
|---|---|---|---|---|
| PPO | 90.4 | 16.0 | 7.5 | 93.9 |
| SAC | 89.7 | 73.2 | 60.3 | 93.2 |
| A2C | 37.1 | 1.8 | 1.8 | 82.0 |
| DQN | 94.1 | 48.4 | 12.9 | 95.1 |

**PPO and A2C collapse under drought/heatwave** (retaining under 20%
of their own normal-regime yield). **SAC generalizes best** (67-82%
retention). These differences are statistically significant across
seeds (Mann-Whitney U, p ~ 0.008-0.016), not attributable to noise.
See `results/figures/degradation_yield_mean.png` and
`degradation_water_productivity_mean.png`.

### Water-use efficiency

`results/figures/pareto_{normal,drought,heatwave,excess_rain}.png`
plot yield vs. water applied per method, per regime. SAC and DQN sit
on or near the efficient frontier at moderate water use across every
regime; brute-force baselines sit at the high-water, marginal-gain
extreme.

### Real-weather comparison against observed practice

Using real historical weather (not synthetic) paired with a
threshold-rule-derived "observed practice" irrigation series (see
below for why it's derived rather than a real farmer's log), tested
at two trigger settings to check the result isn't an artifact of one
arbitrary parameter choice:

| Method | Trigger=0.40 yield / water | Trigger=0.55 yield / water |
|---|---|---|
| observed_practice | 41.5 / 330mm | 60.5 / 390mm |
| SAC | 87.4 / 509mm | 87.4 / 509mm |
| PPO | 89.2 / 614mm | 89.2 / 614mm |
| DQN | 91.5 / 619mm | 91.5 / 619mm |

**Finding: RL uses more water than observed practice at both trigger
settings, but achieves 27-50 points higher yield.** This is a
yield-for-water trade-off, not a water-saving result -- reported here
as-is rather than overstated.

### A2C instability (an unplanned but real finding)

A2C's low average yield isn't uniform weakness -- it's bimodal: 2 of
5 seeds converge to a working policy (yield 82-87), while 3 seeds
collapse to near-zero irrigation (yield 1.9-10.7, one seed applying
literally 0mm across all evaluation episodes). See `pct_negligible_water`
in `results/tables/full_results.csv` and the flat green line in
`results/figures/learning_curves.png`.

## Setup

```bash
python -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell
# source venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
```

## Project layout

```
envs/           IrrigationEnv, weather regimes, discrete-action wrapper for DQN
baselines/      Fixed, threshold, random, and observed-practice (real-data replay) policies
training/       Config + one training script per algorithm (early-stopping by default)
models/         Saved checkpoints: models/{algo}/seed_{n}/  (created after training, not tracked in git)
evaluation/     Rollout logic, metrics, cross-regime evaluation loop, significance tests
data/           Real-weather loader, Open-Meteo CSV parser, observed-practice derivation script
plotting/       Degradation plot, Pareto plot, learning curves
results/        tables/ (CSV) and figures/ (PNG) -- the source of every number above
```

## Reproducing these results

```bash
# Train everything (5 seeds x 4 algorithms), early-stopping on
python -m training.train_all

# Evaluate all trained models + baselines across all synthetic regimes
python -m evaluation.evaluate_all --n-episodes 20

# Statistical significance between algorithms, per regime
python -m evaluation.significance

# Figures
python -m plotting.degradation_plot
python -m plotting.pareto_plot --regime drought
python -m plotting.learning_curves
```

Or run the whole pipeline in one command: `python run_experiment.py --n-episodes 20`.

### Adding the real-weather comparison

```bash
# 1. Download real daily weather (Open-Meteo, free/no signup), save to data/raw/
# 2. Derive a plausible irrigation series from it
python -m data.build_observed_practice \
    --weather-csv data/raw/<your_file>.csv \
    --out data/raw/season_with_irrigation.csv --trigger 0.40

# 3. Evaluate against it
python -m evaluation.evaluate_all --n-episodes 20 \
    --real-weather-csv data/raw/season_with_irrigation.csv \
    --season-start 2023-01-01 --season-end 2023-05-30
```

**Note:** `--trigger` is a free parameter, not measured from data --
always test more than one value (e.g. 0.40 and 0.55) rather than
reporting a single unvalidated setting, as shown in the results above.

## Training termination

Training uses `StopTrainingOnNoModelImprovement` rather than a fixed
timestep target -- each run stops once evaluation reward plateaus for
`EARLY_STOP_PATIENCE` rounds. In practice, PPO and A2C converge around
~500k timesteps while SAC and DQN use the full available budget
(up to 2M). Use `training/train_all_fixed.py` for a fixed, identical
timestep budget across all algorithms instead.

## Known limitations

- Single-bucket soil-water model, not a process-based simulator (DSSAT/APSIM).
- Single synthetic crop, single site.
- A2C's bimodal instability is based on 5 seeds -- more seeds would
  help confirm whether this is a structural failure mode or coincidence.
- Results reflect a single training run per configuration; the
  training-budget effect has not been checked against independent
  replicate runs (`training/config.py:IRRIGATION_RUN_TAG` and
  `evaluation/merge_replicates.py` support running this check).
- The observed-practice benchmark is rule-derived from real weather,
  not an authentic recorded field log.
