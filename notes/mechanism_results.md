# Part 2a: mechanism diagnostics (exploratory)

Run: src/run_mechanism.py, epsilon 0.1, 5 splits x 3 seeds, paired against
non-private. Sanity check reproduces part 1 (+0.067, CI 0.041 to 0.093).
Output: results/mechanism_output.txt.

Predictions: notes/predictions.md, commit 34bc0ff (16:00 on 2026-10-22).
Commit 739c7b6 (16:04) added clarifications on how P2 and P3 are computed;
it changed no threshold and no direction. Both commits precede the results
commit 31c076e. The clarifications were written after the script was
drafted and before it was run.

## Results
- P1 met. Among true positives, mean logit change (DP - non-private):
  men -0.048, women -0.665. Difference -0.617 (CI -0.655 to -0.579),
  same direction in all 5 splits.
- P2 met. With per-group thresholds matched to each group's own
  non-private selection rate, the recall-gap change is +0.008
  (CI -0.002 to 0.017). Diagnostic only: it uses sex at decision time.
- P3 not met. The Male-minus-Female weight difference fell from 0.618 to
  0.301 (positive change in 2 of 15 runs; mean change -0.317). The
  prediction was wrong.

## What this supports
The widening is mostly a shift in where women's true positives sit
relative to a fixed threshold, not a loss of within-group ranking.
A direct distortion of the sex weight is not the explanation.

## What it does not show
- The cause. Clipping versus noise, and group size versus base rate,
  are untested.
- Anything beyond epsilon 0.1, this model, and this dataset.
- Behavior of true negatives (not yet checked).

## Caveats
Exploratory follow-up. The five splits overlap, so intervals are probably
somewhat narrow. P3 rests on 15 runs.

## Part 2b: true negatives (exploratory)

Predictions: notes/predictions.md, commit ab0074f, before the script was
written or run. Script: src/run_negatives.py. Output:
results/negatives_output.txt. Sanity check reproduces part 2a P1
(men -0.048, women -0.665).

- Mean logit shift among true negatives (DP - non-private): men -1.051
  (CI -1.214 to -0.887), women -1.220 (CI -1.295 to -1.145).
- P4 not met: women's negatives shifted far more than their positives
  (-0.665), not within 0.15 of it.
- P5 not met: women's negatives did not stay near zero.
- Verdict: outside both pre-specified bands. The prediction framework
  assumed a single group-wide offset and was too narrow.
- Not pre-specified: relative to their own negatives, women's positives
  moved up by 0.555 (CI 0.483 to 0.626); men's by about 1.0.

## Reading (exploratory)
The shift is not a women-specific offset. Negatives fell by over a logit
in both groups; positives fell less, and men's positives fell much less
than women's. Mean logits mix offset and scale; this was not separated.

## Not shown
The cause; whether the DP model's score scale changed; other epsilons.

## Part 2c: logit scale (exploratory)

Predictions: notes/predictions.md, commit 5f9eed6, before the script was
written or run. Script: src/run_scale.py. Output: results/scale_output.txt.
Sanity check reproduces part 2a P1 (men -0.048, women -0.665).

- SD ratio (DP / non-private) by cell: positives men 1.027, women 0.998;
  negatives men 1.357 (CI 1.277 to 1.438), women 1.234 (CI 1.185 to 1.283).
- P6 not met (mixed): spread expanded for negatives only.
- d' change (DP - non-private): men +0.184 (CI 0.145 to 0.224), women
  +0.124 (CI 0.073 to 0.175); rose in all splits for both groups.
- P7 not met: predicted a fall; d' rose. The difference between groups
  (0.06) was within the stated 0.1.

## Reading (exploratory)
The DP model's negatives are lower and more spread out; positives change
little in spread. d' rising while AUC fell slightly means d' is not a
reliable separation measure here (unequal variances, non-normal scores).
The mechanism is still not identified.

## Pre-specified predictions that failed
P3, P4/P5 (outside both bands), P6, P7. The simple offset and scale
stories did not hold.