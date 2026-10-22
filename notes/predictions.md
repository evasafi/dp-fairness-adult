# Mechanism checks (part 2): predictions fixed before running

Written 2026-10-22, before any of the analyses below were run. The commit history
timestamps this file. Any later change goes under "Deviations" at the bottom; the
predictions themselves are not edited.

## Background (part 1, see README)
At epsilon 0.1 the male-minus-female recall gap widened by about +0.067 across five
splits. Within-group AUC fell by about the same amount for both groups (0.011 men,
0.012 women), and matching the overall selection rate left a +0.064 widening.

## Hypothesis under test (offset)
Under tight DP, women's scores shift down relative to men's with respect to the
decision threshold, while within-group ranking is preserved about equally. This is a
hypothesis about what changes, not an explanation of why.

## Setup (fixed)
- Same five splits (random_state 1 to 5) and seeds 0 to 2 as `run_multisplit.py`.
- Models: non-private and DP at epsilon 0.1. Everything else unchanged (linear model,
  clip 1.0, delta 1e-5, lr 0.01, 10 epochs).
- Every quantity is paired (DP minus non-private) within (split, seed), averaged over
  seeds within each split, then summarised with a 95% t-interval across the 5 splits.
- All runs are reported, including those that contradict the predictions.

## P1: positive-class logits
Quantity: mean model logit among true positives in the test set, per group.
Shift_G = DP minus non-private. Prediction: Shift_F is more negative than Shift_M, and
(Shift_F - Shift_M) is negative in all five splits with a 95% interval excluding zero.
Falsified if the interval includes zero or any split has the opposite sign.

## P2: per-group threshold matching (diagnostic only; it uses sex at decision time)
For each DP model, choose separate thresholds for men and women so that each group's
selection rate equals its own non-private selection rate at threshold 0.5. Then compute
the recall gap. Prediction: the mean change in the gap versus non-private is below 0.02
(it was +0.067 at threshold 0.5). Falsified if it is above 0.04. Between 0.02 and 0.04:
inconclusive, reported as partial.

## P3: the sex coefficients
The one-hot columns sex_Male and sex_Female are collinear with the bias, so only the
difference w_Male - w_Female is meaningful. Prediction: (w_Male - w_Female) under DP
minus non-private is positive in all 15 (split, seed) runs, and its magnitude exceeds
the seed-to-seed spread. Falsified if signs vary or the change is within the spread.
A failed P3 does not refute the offset hypothesis, since correlated features
(for example marital status) could carry the same shift.

## Not tested here
Why the shift occurs; other datasets or attributes; the effect of removing sex from
the features (planned as a separate ablation, with its own predictions).

## Clarifications (written before running, after the script was drafted)
- P2: group-specific thresholds use each group's selection rate at threshold 0.5 from the
  non-private model of the same (split, seed), computed over all test examples of that
  group. The baseline for comparison is the non-private recall gap at threshold 0.5.
- P3: "seed-to-seed spread" means the pooled within-split standard deviation (across the
  3 seeds) of the DP-minus-non-private change in (w_Male - w_Female). P3 is met if all 15
  changes are positive and their mean exceeds that spread.
- The script labels P1, P2 and P3 as MET / NOT MET / INCONCLUSIVE from these rules.

## Deviations
(none yet)

## Part 2b predictions (true negatives; written before running)
Quantity: mean logit shift among TRUE NEGATIVES by group (DP minus
non-private), epsilon 0.1, 5 splits x 3 seeds, same pairing as P1.

- P4 (group-wide offset): women's true-negative shift is within 0.15 of
  their true-positive shift (-0.665), i.e. between about -0.82 and -0.52,
  with the 95% interval across splits excluding zero. Predicted because
  within-group AUC fell about equally for men and women.
- P5 (positive-specific): if women's true-negative shift is above -0.20
  (closer to zero), the shift is specific to positives and the AUC
  argument needs rethinking.
- Anything between -0.52 and -0.20 is reported as mixed, not as support
  for either.
- The thresholds are arbitrary choices, set before running. Exploratory.

## Part 2c predictions (logit scale; written before running)
Quantity: standard deviation of logits within each group x class cell
(men/women x positives/negatives), non-private vs DP, epsilon 0.1,
5 splits x 3 seeds, same pairing as P1.

- P6 (scale expansion): the ratio SD_DP / SD_NP is above 1.2 in all four
  cells, with the 95% interval across splits excluding 1. Ratios between
  0.8 and 1.2 in most cells are reported as "scale about unchanged";
  ratios below 0.8 as "compression". Anything mixed is reported as mixed.
- P7 (separation in SD units): d' = (mean positive logit - mean negative
  logit) / pooled within-class SD falls under DP for both groups, and the
  fall differs between men and women by less than 0.1. Predicted because
  within-group AUC fell by about the same amount in both groups.
- Thresholds (1.2, 0.8, 0.1) are arbitrary choices, set before running.
  Exploratory.

  ## Part 3 predictions (noise vs clipping; written before running)
Design: clipping norm 1.0; noise multiplier = k x sigma*, k in {0, 0.25,
0.5, 1.0}, where sigma* is the multiplier giving epsilon 0.1. 5 splits x
3 seeds, paired against non-private. Quantity: change in the TPR gap at
threshold 0.5 (DP - non-private), averaged over seeds, then across the
5 splits. Epsilon is not matched across arms.

- P8 (clipping alone contributes little): at k = 0 the mean gap change is
  below 0.02. Above 0.04 means clipping contributes substantially.
  Between 0.02 and 0.04 is reported as inconclusive. With a CI half-width
  of about 0.026, a mean below 0.02 is weak evidence, not proof, that
  clipping contributes little.
- P9 (dose-response): mean gap change is non-decreasing across
  k = 0, 0.25, 0.5, 1.0, and the k = 1.0 mean exceeds the k = 0 mean by
  more than 0.04.
- P10 (sanity): the k = 1.0 arm reproduces part 1 (+0.067, CI 0.041 to
  0.093). If it does not, the arms are not comparable and nothing else is
  interpreted.
- Thresholds (0.02, 0.04) are arbitrary choices, set before running.
  Exploratory.