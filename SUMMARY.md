# DP-Adam and the gender recall gap on UCI Adult: summary of parts 1 to 3, 5 and 6

## Question
Does training with differential privacy (DP-Adam) widen the male-minus-female
recall (true positive rate) gap for an income classifier on UCI Adult, and if
so, what drives it? This is a small empirical study. The effect itself is
known (Bagdasaryan et al., 2019: DP hurts underrepresented groups more);
this is a check in a new setting (tabular data, linear model, recall gap).

## Design
Linear classifier on 105 features (48,842 rows), DP-Adam with Opacus
(delta 1e-5, clipping norm 1.0). Five train/test splits x 3 seeds, each
private model paired against the non-private model on the same split and
seed; 95% t-intervals across splits. Part 1's five-split design was set
after an exploratory look at one split, where the question arose. Parts 2,
3 and 5 are exploratory follow-ups with predictions committed before
running (notes/predictions.md). Part 4 (a second dataset) is pre-registered
but has not been run; part 5 was run first. Part 6 (an optimizer comparison)
was pre-registered, including tuning grids, before any tuning or final run.

## Findings
1. At epsilon 0.1 the gap widens by +0.067 (CI +0.041 to +0.093), in all five
   splits. Women's recall falls by 0.104 and men's by 0.037; accuracy falls
   only 0.008. At epsilon 0.5, 1 and 8 no change was detectable (effects up
   to about 0.025 cannot be ruled out with 5 splits).
2. The widening is a position effect. Among true positives, women's mean
   logit falls by 0.665 and men's by 0.048. With group-specific thresholds
   the gap change shrinks to +0.008 (CI -0.002 to +0.017). This is a
   diagnostic only, since it uses sex at decision time.
3. Three simple explanations failed their pre-specified tests: a distorted
   sex weight (the Male-Female weight difference fell, not rose), a
   women-only score offset (true negatives fell by 1.05 for men and 1.22 for
   women), and a uniform change in score spread (spread grew for negatives,
   not positives).
4. The widening grows with noise. At clipping norm 1.0 and noise at 0, 0.25,
   0.5 and 1.0 times the epsilon-0.1 level, the gap change is +0.001, +0.011,
   +0.032 and +0.067. Clipping alone is not detectably responsible, though
   its interval reaches +0.020.
5. Controlled resampling (training size fixed at 12,000, epsilon 0.1, noise
   multiplier 22.5 in every arm): the pre-specified group-size hypothesis
   was contradicted (15% women gave a gap change of +0.082, 50% women
   +0.136; contrast -0.055, 97.5% CI -0.089 to -0.021), and the base-rate
   contrast was inconclusive (+0.033, CI -0.016 to +0.081). Post hoc and
   untested: the DP model's own gap is similar across group-size arms and
   falls when women's base rate is raised (0.282 for the reference, 0.095
   at a rate of 0.25) or men's is lowered (0.004).
6. Optimizer comparison at epsilon 0.1 (part 6; learning rates tuned by
   validation log-loss, not by the gap). DP-Adam at its tuned lr 0.01 (the
   part 1 setting) still widens the gap by +0.067. DP-AdamBC (Tang et al.,
   2024) widened it more (+0.101; contrast with DP-Adam +0.035, 97.5% CI
   +0.017 to +0.052), and DP-SGD with momentum slightly more than DP-Adam
   (+0.077; contrast +0.010, CI +0.007 to +0.013). Two of three
   pre-registered predictions failed (notes/part6_results.md). DP-AdamBC
   needed a large floor to train at all, and about 74% of its
   parameter-steps sat at the floor, so it is only a partial test of the
   noise correction.
## Limitations
One dataset, one linear model, one binary sex attribute. The five splits
overlap (about 30% of any two test sets), so intervals are probably somewhat
narrow. Epsilon is not matched across the noise arms. Parts 2, 3 and 5 were
chosen after seeing earlier results. The DP guarantee covers training only;
preprocessing was fitted without privacy. Part 5's arms change overall class
balance as well as group share, and the non-private gap differs across arms,
so its change metric is not comparable across arms. Its numbers are not
comparable to part 1's (smaller training set, larger noise multiplier).
The private models use Adam as the optimizer (Opacus clipping and noise
wrapped around Adam), not plain SGD, so results may not carry over to
DP-SGD and the comparison with Bagdasaryan et al. (2019) is loose.
Part 6's arms are not matched on test accuracy, and its tuning split
overlaps the test sets of splits 1 to 5.

## Not established
Why noise hurts women's true positives more. Whether clipping and noise
interact (clipping norm was not varied in part 3). Whether group size or
base rate drives the effect (part 5 does not separate them from class
balance). Whether any of this holds on other datasets, models, attributes
or epsilons (part 4 is planned but not run). Whether a clean test of the
DP-AdamBC noise correction, one that is not mostly at its floor, changes
the gap. Whether rare features behave like rare classes in the sense of
the heavy-tail class-imbalance literature: part 6 gives no support for
this, but it does not refute it.

## Reproduce
See README.md. Scripts: src/run_sweep.py, run_multisplit.py, run_clip.py,
analyze.py (part 1 numbers), run_mechanism.py, run_negatives.py,
run_scale.py, run_noise.py, run_resample.py; part 6: adambc.py (DP-AdamBC),
tune_part6.py, tune_part6_sgdm_ext.py, run_part6.py, analyze_part6.py.