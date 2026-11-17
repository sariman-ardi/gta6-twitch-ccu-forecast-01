# GTA VI launch-day Twitch forecaster

Forecasts how many people will be watching Grand Theft Auto VI live on
Twitch in each hour of its release day (19 November 2026, UTC), given
only the first hour's viewer count. Entry for make.fun's bounty
[gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
— total prize $250, 20 places paid.

## How it works

`main.py` (Python 3.13, standard library only, deterministic, no network)
reads one case (`/input/<id>.in`) and writes `/output/<id>.out`: for each
of hours 2–24, five quantiles (5%, 25%, 50%, 75%, 95%) of that hour's
concurrent viewers in the GTA VI Twitch category.

The forecast is anchored on the case's hour-1 print and built from the
38–42 comparable past big launches shipped in the case:

- For each comparable, the log-ratio profile `ln(viewers_h / viewers_1)`
  for hours 2–24 is computed.
- Launch size drives the shape: a launch whose hour 1 is already huge
  grows little later, while a small hour 1 grows a lot (the correlation
  of `ln(hour-1 viewers)` with the log-ratio reaches about −0.9 by hour
  24). So per hour a least-squares fit `l_h = a + b·ln(p1)` is made and
  the five quantiles are taken from the fit's residuals — the
  pinball-optimal shape on the log scale.
- GTA VI unlocks at midnight local time, rolling by country, like the
  practice launches, so the fit is done twice — once on the
  midnight-local comparables only, once on all of them — blended 70/30
  in log space and lightly smoothed across neighbouring hours. The
  median is clamped to the range the pool actually saw, so a record-size
  hour 1 cannot extrapolate the curve to absurd values.
- The GTA VI category's own pre-release history is not a launch curve
  and is excluded from the pool. If too few comparables survive, the
  program falls back to the all-pool fit, then to the house persistence
  baseline, and never crashes on a case.

## Local validation

There is no local grading for this bounty (the server grades every
entry, and the practice answers are held by it), so the model was
validated by leave-one-out cross-validation over the 37 comparable
launch curves in the kit, scored with the contest's pinball loss on the
log scale:

| model | LOO, all 37 launches | LOO, 12 midnight-local launches |
|---|---:|---:|
| this model | 0.556 | **0.243** |
| house persistence baseline | 0.930 | 0.357 |

The midnight-local column is the practice-like subset (the final is a
midnight-local launch); the baseline scores about 0.387 on the kit's
real practice (per the kit's RULES.txt). Runtime is about 0.12 s per
case (limit 8 s), and repeated runs are byte-identical.

## Files

- `main.py` — the entry point (the whole program).
- `LICENSE` — MIT.
- `README.md` — this file.

Run locally:

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Development and test scripts live in `dev/` (validation, leave-one-out
tests, and the downloaded kit in `kit/`); they are not part of the entry.
