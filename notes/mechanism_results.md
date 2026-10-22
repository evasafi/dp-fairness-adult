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