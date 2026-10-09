# Pipeline

Eight notebooks, four datasets. Each notebook is standalone: it reads its inputs as
files and writes its outputs as files, so any stage can be re-run on its own.

| # | Notebook | Runtime | Runs | Reads | Writes |
|---|---|---|---|---|---|
| 01 | `01_data_preprocessing` | CPU | once per dataset | raw dataset file | `df_train_<ds>.csv`, `df_test_<ds>.csv`, converted `<ds>.csv` |
| 02 | `02_cwgan` | GPU | once per dataset | `df_train_<ds>.csv` | `cwgan_<ds>_<setting>_seed42_pool.csv`, checkpoint |
| 03 | `03_ecwgan` | GPU | once per dataset | `df_train_<ds>.csv` | `ecwgan_E12345_<ds>_<setting>_seed42_pool.csv`, checkpoint |
| 04 | `04_ecwgan_vae_filter` | GPU | once per dataset | `df_train_<ds>.csv`, checkpoint of 03 | `ecwganVAE_<ds>_<setting>_seed42_pool.csv`, `..._filterlog.csv` |
| 05 | `05_smotenc` | CPU | once per dataset | `df_train_<ds>.csv` | `smotenc_<ds>_nmax<N>_seed42.csv` |
| 06 | `06_synthetic_quality` | CPU | once | every `df_train`, `df_test`, `*_pool.csv` | `quality_all.csv` |
| 07 | `07_classification` | CPU | once | every `df_train`, `df_test`, `*_pool.csv`, `smotenc_*.csv` | `results_*.csv`, `budget_rule_sensitivity.csv` |
| 08 | `08_settings_search` | GPU | once per dataset | converted `<ds>.csv` from 01 | `search_trials_<ds>.csv`, `search_selected_<ds>.json` |

`<ds>` is one of `diabetes`, `pageblocks`, `yeast`, `ecoli`.
`<setting>` is `u10000_b250_lr2e-04_z128`. `E12345` in a file name means that all
five components E1–E5 are on; it is the CA-CWGAN (E5) arm of the paper. `ecwgan` in every
file, checkpoint and scenario name refers to CA-CWGAN (`ecwganVAE` = CA-CWGAN with E6).

Order: 01 → 02 → 03 → 04 → 05 → 06 → 07. Notebook 08 depends only on notebook 01's
converted CSV and can run in parallel with 02–07.

---

## Step 1 — Data preprocessing (`01`)

1. Raw UCI files of Yeast, Ecoli and Page Blocks (headerless, whitespace-separated)
   are converted to CSV with named columns. The identifier column `Sequence_Name` is
   dropped and Page Blocks' numeric labels are given their names.
2. Exact duplicate rows are removed (23,899 on the clinical dataset), then any row
   with a missing value.
3. Classes with fewer than 10 rows are removed (`MIN_CLASS_SIZE`).
4. The target is encoded as 0..K−1 in sorted label order.
5. Stratified 80/20 split, `random_state=42`.
6. `MinMaxScaler` is fitted **on the training rows only** and applied to both sides.
   Hold-out values outside [0, 1] are left unclipped: both classifiers are tree
   ensembles, so a monotone transform cannot move a split point.

| Dataset | Rows after cleaning | Features | Classes | Training IR |
|---|---|---|---|---|
| CDC Diabetes Health Indicators | 229,781 | 21 | 3 | 41.06 |
| UCI Page Blocks | 5,406 | 10 | 5 | 177.77 |
| UCI Yeast | 1,448 | 8 | 9 | 21.88 |
| UCI Ecoli | 327 | 7 | 5 | 7.12 |

A column with at most 12 distinct values (`MAX_DISCRETE_LEVELS`) is treated as
discrete everywhere in the pipeline.

## Step 2 — CWGAN baseline (`02`)

Conditional WGAN, Lipschitz constraint by weight clipping (c = 0.01), 5 critic steps
per generator update, Adam (β = 0.5, 0.9). The class label is drawn at its empirical
frequency (a real row and its label are sampled together). One sigmoid output head;
discrete columns are snapped to their nearest legal level after generation and
continuous columns clipped to the training range.

The pool holds `N_MAX` rows for **every** class (majority included), so every
training-set scenario of step 7 can be cut from it.

## Step 3 — CA-CWGAN with the k-NN filter (`03`)

Same backbone, same setting, plus E1–E5:

- **E1** gradient penalty, λ = 10, computed on packed interpolates.
- **E2** each continuous column is fitted with a `BayesianGaussianMixture`
  (Dirichlet-process prior 1e-3, at most 10 components); a value is encoded as a
  one-hot mode indicator plus its offset `(v − μ) / 4σ`.
- **E3** the condition is drawn from q(y = k) ∝ n_k^τ with τ = 0 (uniform over classes),
  then a real row of that class.
- **E4a** Gumbel-softmax (temperature 0.2) over the levels of each discrete column and
  over the modes of each continuous column, tanh for the offset.
- **E4b** the critic scores 10 samples jointly (PacGAN); the batch is rounded down to a
  multiple of 10 (250 stays 250).
- **E5** after training, every generated value is clipped to the range observed for
  that column *within that class*, discrete columns are snapped to legal levels, and
  a row of class k is kept when at least 40% of its 5 nearest real training rows also
  carry k. At most half of any generation round is discarded; the pool builder
  over-generates and repeats (≤ 8 rounds) until every class holds `N_MAX` rows.

On large problems (the clinical dataset) the nearest-neighbour search runs on the GPU
(`KNN_BACKEND = "auto"`). The GPU and CPU searches can break exact-distance ties
differently (measured: 0.03% of keep/drop decisions on the clinical dataset, same
number kept). Set `KNN_BACKEND = "sklearn"` to force the CPU path.

The checkpoint written at the end of training also stores the random-number state;
notebook 04 restores it.

**Component switches (ablation).** `USE_E1_GRADIENT_PENALTY` … `USE_E5_QUALITY_CONTROL`
are all `True` for the paper's CA-CWGAN. Turning one off reproduces the original code's
fallback: E1 off → weight clipping (c = 0.01); E2 off → one scaled value per continuous
column; E3 off → empirical class frequency (τ = 1); E4 off → plain softmax heads and no
packing; E5 off → no post-generation filter. The active set is part of the file name
(`ecwgan_E12345_…` is the full model, `ecwgan_E2345_…` has E1 off). To score an ablation
pool in notebook 07, add a pattern such as `"ecwgan_E2345": "ecwgan_E2345_{ds}_*_pool.csv"`
to `POOL_PATTERNS`. The ablation was not run for the paper.

This notebook also draws diagnostic plots (class distribution and composition of the
augmented set, and real vs synthetic per-column
histograms and boxplots on `min class size` rows per class; Page Blocks in the paper).

## Step 4 — CA-CWGAN with the VAE filter, E6 (`04`)

No generator is trained. The generator of notebook 03 is loaded from its checkpoint,
the random state saved at the end of its training is restored, and the pool is rebuilt
with the k-NN vote replaced by a density test:

- one small VAE per class (2 × 64 hidden units, latent 8, 200 epochs, batch 128,
  lr 1e-3) is fitted on the real training rows of that class;
- after the E5 range alignment, a generated row labelled k is kept when the class-k VAE
  gives it a lower negative ELBO than every other class's VAE (`VAE_MARGIN = 0`);
- a class with fewer than 30 real rows (`MIN_VAE_ROWS`) gets no VAE: its generated rows
  are kept after the range alignment without a density test, and that class is not
  used as a rival for the others;
- the VAE has no drop cap, so each round requests rows according to the acceptance
  rate observed so far (at least 256, at most 8 × the shortfall + 256, ≤ 8 rounds).

`..._filterlog.csv` records, per class, the rows kept, generated and the survival rate.

## Step 5 — SMOTENC control (`05`)

Every minority class is raised to `N_MAX` with SMOTENC (`k_neighbors = min(5,
smallest class − 1)`), columns with ≤ 12 levels declared nominal. Page Blocks has no
such column, so plain SMOTE is used there (SMOTENC requires at least one nominal column).

## Step 6 — Synthetic data quality (`06`)

For every pool against the real training rows:

- **Fidelity** on the pool cut to the real class counts (so both sides share the same
  class mix), 500 rows per class: mean Kolmogorov–Smirnov statistic and mean
  Wasserstein distance over the continuous columns; mean absolute difference of the
  correlation matrices (constant columns skipped).
- **Privacy proxies**: 5th percentile of the distance to the closest real record (DCR),
  on 2,000 rows per side, read against the same DCR between the training and hold-out
  rows (the real-vs-real reference); and the number of generated rows whose features
  exactly equal a training row, counted over the **full** pool.

In the paper, DCR is reported for the CWGAN and CA-CWGAN (E5) pools; the identical-row
count is zero for all twelve pools (four datasets × CWGAN, E5, E6).

## Step 7 — Classification and evaluation (`07`)

Every scenario is fitted on its own training set and scored on the same real hold-out.

| Scenario | Training set |
|---|---|
| `baseline` | real rows |
| `class_weight` | real rows, classifier with `class_weight="balanced"` |
| `smotenc` | SMOTENC output of step 5 |
| `<g>` | real rows + pool rows topping every class up to `N_MAX` (augmented) |
| `<g>_syn` | pool only, `N_MAX` per class (synthetic only) |
| `<g>_synmatch` | pool only, at the real class counts (synthetic only, matched) |
| `<g>_ratio3` | real rows + pool rows raising every class to `ceil(N_MAX / 3)` (residual IR 3) |
| `<g>_derived` | real rows + pool rows up to the budget-rule targets (derived budget) |
| `<g>_dsyn` | pool only, at the budget-rule targets (derived synthetic only) |

`<g>` is `cwgan`, `ecwgan` (E5) or `ecwganVAE` (E6). Pool rows are always the first n
rows of each class, so every scenario is a deterministic cut of one pool.

**Budget rule** (from the training class counts only): IR ≥ 40 → every class to
`N_MAX`; 10 ≤ IR < 40 → every class to `ceil(N_MAX / 3)`; IR < 10 → no augmentation;
any class with fewer than 40 real rows is capped at 4 × its own count. It selects full
balance on the clinical dataset and Page Blocks, residual IR 3 on Yeast and no
augmentation on Ecoli. Where the rule coincides with an existing scenario
(`_derived` = augmented, or `_dsyn` = `_syn` / `_synmatch`) no second model is fitted.
The notebook also reports what the rule selects over 60 threshold combinations
(`budget_rule_sensitivity.csv`).

**Classifiers** (fixed across all scenarios):

- LightGBM: 300 estimators, learning rate 0.05, max depth 6, 31 leaves,
  `reg_alpha = reg_lambda = 0.1`, `min_child_samples = 20`, seed 42, fitted once
  (deterministic).
- Random Forest: 300 trees, `min_samples_leaf = 2`, seeds 42, 43, 44, averaged.

**Metrics**: accuracy, balanced accuracy, macro F1, geometric mean of per-class
recalls (G-mean), per-class recall. The LightGBM loss curve per boosting iteration is
recorded for the baseline, cost-sensitive and augmented generator scenarios; the
`eval_set` holds training rows only and the hold-out curve is computed after fitting,
so the hold-out never enters `fit()`. No early stopping is used.

Runs are resumable: each finished scenario is appended to `results_long.csv` and
skipped on the next run. Use a fresh `RESULTS_DIR` when the inputs change.

## Step 8 — Equal-budget settings search (`08`)

Starts from notebook 01's converted CSV, repeats steps 1–6 of the preprocessing, and
searches on the training rows only: a stratified 20% validation split is carved out of
them and the search's training part is capped at 20,000 rows.

- Eight candidates are drawn once by `ParameterSampler` (seed 42) from
  batch ∈ {128, 250, 256, 500}, lr ∈ {1e-4, 2e-4, 5e-4}, noise_dim ∈ {32, 64, 128}.
- Every arm — CWGAN (weight clipping), CWGAN-GP (gradient penalty; otherwise identical
  to CWGAN) and CA-CWGAN (E1–E5) — receives the same eight candidates and is trained for
  2,000 updates each.
- Score: macro F1 on the validation rows of a LightGBM (150 trees, learning rate 0.08,
  31 leaves) trained on up to 2,000 synthetic rows per class only.
- The best candidate of each arm is written to `search_selected_<ds>.json`; the spread
  (max − min) over the eight candidates is printed.

---

## Where each result in the paper comes from

| Paper | Source |
|---|---|
| Figure 1 | research workflow diagram (in the paper only) |
| Figure 2 | CA-CWGAN architecture diagram (in the paper only) |
| Table 2 | notebook 01 output |
| Table 3 | configuration of notebooks 02, 03, 04 (equivalent epochs printed by 02/03) |
| Table 4, Table 6, Section 3.1–3.2 | notebook 07: `results_summary.csv`, `results_long.csv` |
| Table 5, Section 3.3 | notebook 08: `search_trials_<ds>.csv`, `search_selected_<ds>.json` |
| Section 3.4 (fidelity, privacy) | notebook 06: `quality_all.csv`; validity of discrete cells printed by 02/03 |
| Figure 3 | `figures/make_fig3_per_class_recall.py` from `results_long.csv` (notebook 07 also draws a colour version) |
| Figure 4 | `figures/make_fig4_leading_confusion.py` from `results_confusion.csv` (LightGBM leaders of Table 6) |
| LightGBM loss curves (mentioned in Section 3.5, not shown) | notebook 07: `results_curves.csv` |

## Run time (Colab T4)

Training is counted in generator updates, not epochs, so a run costs about the same on
every dataset: about 9 minutes per 10,000 CA-CWGAN updates on a T4. Pool building on the
clinical dataset adds several minutes (E5 filters about 912,000 candidate rows against
183,824 real ones). Notebooks 02, 03 and 08 checkpoint and stop cleanly after
`MAX_MINUTES = 150`; re-run the same notebook unchanged to continue. Point
`CHECKPOINT_DIR` at Google Drive so a disconnected session does not lose the checkpoint.
