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

## Part 4 predictions (second dataset; written before running)
Data: ACS Income via folktables, California, 2018 1-year person survey,
default ACSIncome definitions (label: income above $50,000). 10 features,
categorical ones one-hot encoded with categories under 0.5% of training
rows grouped into one bucket; AGEP and WKHP numeric; sex kept as a
feature. Training set subsampled to 34,189 rows to match Adult; test set is
the full 30% split. Epsilon 0.1, delta 1e-5, clipping norm 1.0, linear
model, 5 splits x 3 seeds, paired against non-private; same gap definition
as part 1 (TPR gap at threshold 0.5, male minus female).

- P11 (attenuation): mean gap change is below +0.04 (part 1 on Adult:
  +0.067). Predicted because women are not a rare, low-base-rate group
  here (positives: men 46.6%, women 34.9%). At or above +0.04 with the
  interval excluding zero counts as replication.
- P12 (direction): women's recall change minus men's is negative in at
  least 4 of 5 splits.
- P13 (sanity): the non-private AUC is above 0.85 in all splits. If not,
  nothing else is interpreted.
- A replication would suggest the effect is not specific to Adult; an
  attenuation would be consistent with a group-structure explanation but
  would not isolate it, since the datasets differ in many ways. The
  thresholds (0.04, 0.85) are arbitrary choices, set before running.
  Exploratory.

## Part 5 predictions (controlled resampling in Adult; written before running)
Note: part 5 runs before part 4. The part 4 predictions above stay as
committed and part 4 is still to be run.

Design: for each of the 5 splits (same stratified 70/30 splits as part 1),
the TRAINING set is resampled to N = 12,000 rows by drawing within each
sex-by-label cell, so that features within a cell are unchanged. The TEST
set is the full natural 30% split in every arm. Linear model, DP-SGD
epsilon 0.1, delta 1e-5, clipping norm 1.0, 3 seeds per split. For each
(split, arm) one resampled training set is drawn with a fixed seed; the
non-private model is trained on that same resampled set. Gap = TPR gap at
threshold 0.5, male minus female. Gap change = DP gap - non-private gap,
averaged over seeds, then across splits.

Arms (women's share of training data / women's positive rate / men's
positive rate):
  Ref: natural share (about 33.2%) / 0.110 / 0.304
  A1: 15% / 0.110 / 0.304      A2: 50% / 0.110 / 0.304
  B1: natural / 0.20 / 0.304   B2: natural / 0.25 / 0.304
  B3: natural / 0.110 / 0.110
Counts are rounded to whole rows. Before any training the script checks
that every cell requested by every arm is available in all 5 splits, and
stops if not.

Hypotheses (my guesses; the data may contradict them):
- P14 (floor check): if the Ref arm's mean gap change is below +0.02, the
  contrasts below are not interpreted; the result is reported as "no
  detectable widening at N = 12,000".
- P15 (group size, primary contrast C1): C1 = (gap change in A1) minus
  (gap change in A2) is positive, with a 97.5% t-interval across splits
  that excludes zero. A negative value with an interval excluding zero
  contradicts the hypothesis.
- P16 (base rate, primary contrast C2): C2 = (gap change in B2) minus
  (gap change in Ref) is negative, with a 97.5% interval excluding zero.
- P17 (descriptive): the mean gap changes order as A1 >= Ref >= A2 and
  Ref >= B1 >= B2. Reported, not tested; B3 is exploratory with no
  predicted direction.
- C1 and C2 are the only pre-specified tests; the 97.5% intervals adjust
  for two comparisons. Everything else is exploratory. With 5 overlapping
  splits the intervals are wide, so an interval that includes zero is
  inconclusive, not evidence of no effect.
- The Ref arm at N = 12,000 will not necessarily reproduce part 1's
  +0.067, because the noise per update differs at this size. Contrasts are
  always against Ref at the same N.
## Correction (2026-11-14): optimizer name
The sections above call the private training method "DP-SGD". The models
use Adam (torch.optim.Adam, lr 0.01) with Opacus per-example clipping and
Gaussian noise added to the gradients, so the method is DP-Adam. The
predictions, thresholds and designs above are unchanged; only the name is
corrected. Whether the results carry over to plain DP-SGD has not been
tested. The README, SUMMARY and source docstrings were corrected in
commits a243592, b878947 and 1502da0.
## Part 6 (written before any tuning or running): does the optimizer matter?

Written 2026-11-14. No part 6 tuning or run has been done.

Why: part 1 found a wider male-minus-female recall gap at epsilon 0.1. The
private runs use Adam (DP-Adam) at lr 0.01, which was not tuned. Tang,
Shpilevskiy and Lecuyer (AAAI 2024) argue that at high noise DP-Adam behaves
like DP-SGD with momentum, because the noise biases Adam's second-moment
estimate, and propose DP-AdamBC to remove that bias. Tang, Zhiyanov and
Lecuyer (HiLD 2025) study heavy-tailed class imbalance and DP optimizers.
Untested idea (mine, not from those papers): rare features, such as those
that identify women's true positives, play the role of rare classes.

Arms (same noise multiplier, so the same epsilon, in every arm):
A. DP-Adam (as in parts 1 to 5)
B. DP-AdamBC (src/adambc.py, written from the paper; matches torch Adam
   exactly at zero noise, see src/check_adambc.py)
C. DP-SGD with momentum 0.9

Fixed: linear model, 10 epochs, batch size 512, clipping norm 1.0,
delta 1e-5, epsilon 0.1, splits 1 to 5, seeds 0 to 2, non-private baseline
as in part 1.

Tuning (grids fixed now): learning rate per arm on a validation holdout
(20 percent of the training part of split 0), chosen by validation log-loss,
never by the gap. A and B: lr in {0.001, 0.003, 0.01, 0.03, 0.1}. C: lr in
{0.03, 0.1, 0.3, 1, 3}. B also: gamma in {1e-8, 1e-6, 1e-4, 1e-3}. If the
best value sits at a grid edge, extend the grid by one step and say so.
Limitation: split 0 rows also appear in the test sets of splits 1 to 5, so
tuning is not fully separate from testing. Arm A is also reported at lr 0.01
(the part 1 setting).

Outcome: change in male-minus-female recall gap relative to the non-private
model (as in part 1). Secondary: women's recall, men's recall, accuracy.
Contrasts B minus A and C minus A, with 97.5 percent t-intervals across splits.

Predictions (edit before committing so they say what I expect):
P1. Tuned DP-Adam still shows a positive gap change at epsilon 0.1.
P2. B has a smaller gap change than A (contrast below 0). Confidence is
    modest, because at epsilon 0.1 the correction may sit at its floor.
P3. C is close to A (contrast interval contains 0).

What would count against these:
- P1 fails if the tuned arm A interval contains 0 (the part 1 effect then
  depends on the untuned learning rate).
- P2 fails if the B minus A interval contains 0 or lies above 0, or if B
  diverges for every gamma in the grid (reported as a failure, not dropped).
- P3 fails if the C minus A interval excludes 0.

## Part 6 selected hyperparameters (recorded before any final run)

Chosen by mean validation log-loss at epsilon 0.1 (split 0, inner 80/20 split, seeds 0-2), never by the gap.

- Arm A, DP-Adam: lr 0.01 (interior optimum; same value as part 1)
- Arm B, DP-AdamBC: lr 0.003, gamma 1e-4 (gamma 1e-8 was unusable: val log-loss 0.57 to 3.5)
- Arm C, DP-SGD momentum 0.9: lr 0.03 (grid extended downward once; 0.01 and 0.003 were worse, so the optimum is interior)

Best validation log-loss: A 0.353, B 0.348, C 0.355. Differences of about 0.007 are not ranked.

Known limitations of tuning: split 0 training rows overlap test sets of splits 1-5; tuning used 80% of the split-0 training part, so sigma differs slightly from final runs; preprocessing was fitted on the full split-0 training part.

Exploratory addition made after seeing the tuning results (not pre-registered): a diagnostic for the share of AdamBC parameters whose corrected second moment is above gamma, because only large floors worked.

Part 6 outcome (added 2026-11-14, after the final run): P1 held; P2 and P3
failed by the criteria above. See notes/part6_results.md.
