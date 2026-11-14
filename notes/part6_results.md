# Part 6 results: does the optimizer change the gap widening?

Run 2026-11-14. Pre-registration: notes/predictions.md (committed before tuning
and before any final run). Raw results: results/part6.csv. Analysis:
src/analyze_part6.py, output in results/part6_analysis.txt.

Setup: UCI Adult, linear model, epsilon 0.1, splits 1-5, seeds 0-2. Arms:
A DP-Adam lr 0.01; B DP-AdamBC lr 0.003, gamma 1e-4; C DP-SGD momentum 0.9
lr 0.03. Learning rates (and gamma) were chosen by validation log-loss, never
by the gap. Arm A reproduces the stored part 1 rows exactly.

## Results

Change in male-minus-female recall gap vs the non-private model (mean over 5
splits, 95 percent t-interval):

- A DP-Adam:   +0.067 [+0.041, +0.093]
- B DP-AdamBC: +0.101 [+0.070, +0.133]
- C DP-SGD-m:  +0.077 [+0.050, +0.103]

Contrasts (97.5 percent t-interval): B minus A +0.035 [+0.017, +0.052];
C minus A +0.010 [+0.007, +0.013]. B and C are above A in all 5 splits.

Women's recall change: A -0.104, B -0.160, C -0.118. Men's: A -0.037,
B -0.058, C -0.041. Accuracy change: A -0.0079, B -0.0114, C -0.0093.

## Scoring against the pre-registered predictions

- P1 (tuned DP-Adam still widens the gap): held. Interval excludes 0.
- P2 (B widens less than A): failed. The contrast lies above 0, the stated
  failure condition. B widened the gap more than A.
- P3 (C close to A, contrast interval contains 0): failed by the stated
  criterion. The interval excludes 0, although the size is small (0.010).

## What this does and does not show

- The part 1 widening is not an artifact of an untuned learning rate: lr 0.01
  was the validation optimum for DP-Adam.
- In this setup, switching to DP-AdamBC or to momentum SGD did not reduce the
  widening; DP-AdamBC increased it.
- DP-AdamBC is only a partial test of the noise correction. The near-pure
  correction (gamma 1e-8) did not train at epsilon 0.1 (validation log-loss
  0.57 to 3.5). The working setting has a large floor, and on average 73.7
  percent of parameter-steps sat at the floor (min 0.712, max 0.761 across
  runs). There the update is close to momentum SGD with a larger step, so B
  is not a clean test of the correction. This diagnostic was added after
  seeing the tuning results and is exploratory.
- The arms are not matched on test utility. B loses more accuracy than A.
  Recall falls for both men and women in every arm, which looks more like a
  shift in predicted probabilities than a group-specific failure to learn.
  This is a reading of the numbers, not something tested here.
- Exploratory, not pre-registered, three points only: the ratio of gap
  change to accuracy loss was similar across arms (A 8.4, B 8.9, C 8.2).
  If real, the optimizer mainly changes how much the model degrades. Treat it
  as a hypothesis.
- The idea that rare features behave like rare classes (heavy-tail paper) was
  not supported here, but this experiment does not refute it: one dataset, a
  linear model, a binary sex attribute, and the correction was largely at its
  floor. The heavy-tail paper itself studies class frequency, not demographic
  groups, and reports training metrics.

## Limitations

- The 5 splits are resamples of one dataset and overlap, so intervals are
  optimistic (same caveat as part 1).
- Tuning used split 0, whose rows overlap the test sets of splits 1-5; tuning
  used 80 percent of the split-0 training part, so sigma differs slightly
  from final runs; preprocessing was fitted on the full split-0 training part.
- One dataset, one model class, one epsilon, one sensitive attribute.
- Part 4 (second dataset) is still not run.
