# DP-SGD and the gender recall gap on UCI Adult: summary of parts 1 to 3

## Question
Does training with differential privacy (DP-SGD) widen the male-minus-female
recall (true positive rate) gap for an income classifier on UCI Adult, and if
so, what drives it? This is a small empirical study. The effect itself is
known (Bagdasaryan et al., 2019: DP hurts underrepresented groups more);
this is a check in a new setting (tabular data, linear model, recall gap).

## Design
Linear classifier on 105 features (48,842 rows), DP-SGD with Opacus
(delta 1e-5, clipping norm 1.0). Five train/test splits x 3 seeds, each
private model paired against the non-private model on the same split and
seed; 95% t-intervals across splits. Part 1's five-split design was set after an exploratory look at one split, where the question arose. Parts 2 and 3 are exploratory follow-ups with predictions committed before running
(notes/predictions.md).

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

## Limitations
One dataset, one linear model, one binary sex attribute. The five splits
overlap (about 30% of any two test sets), so intervals are probably somewhat
narrow. Epsilon is not matched across the noise arms. Parts 2 and 3 were
chosen after seeing part 1. The DP guarantee covers training only;
preprocessing was fitted without privacy.

## Not established
Why noise hurts women's true positives more. Whether clipping and noise
interact (clipping norm was not varied in part 3). Whether any of this holds
on other datasets, models, attributes or epsilons.

## Reproduce
See README.md. Scripts: src/run_sweep.py, run_multisplit.py, run_clip.py,
run_mechanism.py, run_negatives.py, run_scale.py, run_noise.py.