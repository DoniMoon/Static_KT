# Static_KT

This repository is a paper-facing, cleaned-up version of our Static KT experiments.
It contains only the contribution-specific code:

- `StaticKT`: a static, correlation-based KT model in the log-odds domain
- `IndependentKT`: the masking variant that removes same-KC history items
- Interpretable ensembles between `StaticKT` and `PFA`
  - constant linear mixture
  - sigmoid mixture by user history length `k`
  - sigmoid mixture by KC attempt count `a`

The original benchmark codebase is not vendored here. For the baseline preprocessing,
dataset layout, and sparse `PFA` features, please use the original repository:

- https://github.com/theophilegervet/learner-performance-prediction

## Method

We formulate Static KT in the log-odds domain as follows:

```tex
\begin{equation}
\begin{split}
\mathrm{logit}\,P(q_i = 1 \mid H) = \mathrm{logit}(p_i) \\
+ \sum_{k \in H} \beta_{ik \mid H} \Big( a_k \, \Delta^{+}_{ik} + (1 - a_k) \, \Delta^{-}_{ik} \Big)
\end{split}
\end{equation}
```

Here, `p_i` denotes the prior probability of answering item `i` correctly, capturing
domain-wide statistical biases. The coefficient `beta_{ik | H}` represents the relevance
of a past item `k` to the target item `i`, implemented via an attention mechanism.
The parameters `Delta^{+}_{ik}` and `Delta^{-}_{ik}` capture the positive or negative
evidence provided by the observed outcome `a_k` of item `k`.

Importantly, there is no temporal decay, no forgetting curve, and no explicit learning
transition. The predictive power comes strictly from population-level item correlations.

### Static KT

`StaticKT` is the base model in this repository.

- Prior term: `logit(p_i)`
- Relevance term: attention over previous items
- Evidence term: separate positive and negative evidence embeddings
- No recurrence, no hidden student state transition, no decay

This is the cleaned-up version of the earlier prototype that was previously named
`PriorKT`. In this repository we use the name `StaticKT` to match the paper framing.

### IndependentKT: same-KC masking

`IndependentKT` is the masking ablation of `StaticKT`.

Let `Q_i` be the KC set of target item `i` and `Q_k` the KC set of a history item `k`.
We define the independent-history mask as

```tex
m_{ik} = \mathbf{1}[Q_i \cap Q_k = \varnothing].
```

Only history items with `m_{ik} = 1` are allowed to contribute to attention and evidence.
Equivalently, the model renormalizes attention over the subset of history items that do
not share any KC with the target item.

This isolates the diagnostic effect of cross-KC item correlations:

- same-KC practice effects are masked out
- same-skill retrieval cues are masked out
- only item correlations across disjoint KC sets remain

If no valid independent history remains, the prediction collapses to the item prior.

### Constant ensemble with PFA

We use the simplest interpretable mixture first:

```tex
P_{\mathrm{ens}} = (1 - \alpha) P_{\mathrm{StaticKT}} + \alpha P_{\mathrm{PFA}}.
```

This gives a single global `PFA` weight per dataset.

### Sigmoid ensemble by user history length `k`

To inspect whether explanatory strength shifts with total user history, we use

```tex
\alpha(k) = \sigma(s_k \cdot (k / c_k) + b_k),
```

and then

```tex
P_{\mathrm{ens}}(k) = (1 - \alpha(k)) P_{\mathrm{StaticKT}} + \alpha(k) P_{\mathrm{PFA}}.
```

Here `k` is the number of previous interactions by the same user, and `c_k` is a
dataset-specific scale used only for numerical stability.

### Sigmoid ensemble by KC attempt count `a`

To inspect whether explanatory strength shifts with repeated practice on the target KC,
we use

```tex
\alpha(a) = \sigma(s_a \cdot (a / c_a) + b_a),
```

and

```tex
P_{\mathrm{ens}}(a) = (1 - \alpha(a)) P_{\mathrm{StaticKT}} + \alpha(a) P_{\mathrm{PFA}}.
```

In this repository:

- `a` is the previous attempt count on the target KC(s)
- for multi-KC items, we use the mean attempt count across the target item's KC set

## Expected data layout

This repository assumes the same dataset layout used by the original benchmark:

```text
data/
  <dataset>/
    preprocessed_data_train.csv
    preprocessed_data_test.csv
    q_mat.npz
    X-sscwa.npz
```

Notes:

- `q_mat.npz` is required for `IndependentKT` and for the `a`-based ensemble.
- `X-sscwa.npz` is required for the `PFA` side of the ensemble.
- The ensemble scripts expect `PFA` predictions to be available in `preprocessed_data_test.csv`
  under `LR_sscwa`.
- By default the scripts also expect the original static-model prediction column name
  `BASELINE`, because that is what the benchmark files use.

## Repository layout

```text
Static_KT/
  README.md
  requirements.txt
  static_kt/
    common.py
    models/
      static_kt.py
      independent_kt.py
  scripts/
    train_static_kt.py
    eval_independent_kt.py
    train_constant_ensemble.py
    train_sigmoid_k_ensemble.py
    train_sigmoid_a_ensemble.py
```

## Script guide

### Train StaticKT

```bash
python scripts/train_static_kt.py \
  --data-root /path/to/data \
  --datasets assistments09
```

This trains `StaticKT`, saves a checkpoint, and writes a test prediction file.

### Evaluate IndependentKT

```bash
python scripts/eval_independent_kt.py \
  --data-root /path/to/data \
  --checkpoint-dir results/static_kt/checkpoints \
  --datasets assistments09
```

This loads a trained `StaticKT`, converts it to `IndependentKT` by applying same-KC
history masking, and saves the masked predictions.

### Train constant ensemble

```bash
python scripts/train_constant_ensemble.py \
  --data-root /path/to/data \
  --datasets assistments09
```

### Train sigmoid ensemble by `k`

```bash
python scripts/train_sigmoid_k_ensemble.py \
  --data-root /path/to/data \
  --datasets assistments09
```

### Train sigmoid ensemble by `a`

```bash
python scripts/train_sigmoid_a_ensemble.py \
  --data-root /path/to/data \
  --datasets assistments09
```

## Notes on scope

This repository is intentionally narrow:

- it does not copy the full benchmark
- it keeps only the contribution-specific models and analysis scripts
- it is meant to document the paper contribution cleanly, not to replace the original benchmark

## Attribution

The underlying benchmark and preprocessing pipeline are based on:

- Theophile Gervet et al., learner-performance-prediction
- https://github.com/theophilegervet/learner-performance-prediction
# Static_KT
