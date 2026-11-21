# Project 3 — Agent Handoff and Execution Specification

**Working title:** *Robust Financial Time-Series Classification Under Structured Data Corruption*  
**Subtitle:** A cross-market Python/C++ fault-injection and evaluation framework  
**Course:** FYS-STK3155/FYS4155, Data Analysis and Machine Learning, University of Oslo, Fall 2026  
**Course deadline:** 11 December 2026  
**Specification status:** Working plan; not a completed report, not a statement of results, and not proof of novelty.  
**Owner:** Student author. AI agents are technical assistants, not owners of the research decisions.

---

## 0. Read this first: instructions to any AI agent

You are assisting with a course project and an internship-oriented engineering demonstrator. Treat this document as the current project specification. Read it before proposing changes or writing code.

### Your operating rules

1. **Preserve the central objective.** The project studies the effect of controlled financial time-series data corruption on volatility-transition classification, and builds a reproducible Python/C++ fault-injection and evaluation pipeline.
2. **Do not silently expand scope.** Do not add sector ETFs, many extra indicators, PCA/SVD, wavelets, spectral models, extra neural architectures, autonomous agents, lawyer-allocation models, or a C++ neural-network training stack to the minimum project. Suggest such work only as a clearly separated extension, with an estimate of cost and a reason it is needed.
3. **Inspect before editing.** First inspect the repository tree, current code, data-loader contracts, environment, tests and existing documentation. Reuse working code where appropriate. Do not assume that code mentioned in a conversation is still identical to the current repository version.
4. **Do not invent data facts.** Never assert a date range, instrument count, price field, missingness rate, label frequency, benchmark availability or model result unless it has been observed in the current data or a retained artefact. Distinguish historical observations from facts verified in the current run.
5. **Do not silently repair or reinterpret data.** Any price-field fallback, date alignment, missing-value handling, outlier handling, instrument filtering or target change must be explicit, logged and justified.
6. **Keep ground truth separate from injected corruption.** The clean reference series defines labels. Faults modify input data used to generate test-time features; they must not modify the reference labels.
7. **Protect the test set.** Do not use test results to choose thresholds, features, fault severity, hyperparameters, model architecture or mitigation strategy. If test results have already been inspected, record that fact and do not pretend the test set remains untouched.
8. **Make changes testable and traceable.** Every meaningful change must have an associated test, a recorded rationale, and an updated run manifest or decision log where relevant.
9. **Do not claim that C++ is faster without a fair benchmark.** The C++ component exists primarily to provide a reusable, deterministic, independently tested fault-injection engine. NumPy/pandas already execute much numerical work in compiled code.
10. **Do not claim novelty prematurely.** Fault injection, data-quality-aware modelling, volatility forecasting and regime classification are established topics. The exact project combination may be a useful application, but priority requires a targeted literature review.
11. **Respect the course's LLM policy.** The supplied Project 3 instructions allow generative AI for report text only at contribution level 1 (spelling, grammar and minor phrasing). Do not draft, rewrite or restructure scientific report sections. LLM-assisted code must carry the course-required inline `LLM-assisted` tag and must be tested and understood by the student. Do not fabricate an LLM-use declaration.
12. **Be explicit about uncertainty.** If a decision cannot be made from available evidence, report the uncertainty, the evidence needed and the smallest useful next test. Do not fill gaps with confident guesses.

### Required format for every agent handoff

When you finish a task, report:

- **Objective addressed:** what specification item you worked on.
- **Repository inspected:** relevant files and current versions.
- **Changes made:** exact files and concise description.
- **Tests run:** exact commands, results and any failures. Never say tests passed if they were not run.
- **Evidence produced:** logs, plots, manifests, metrics or other artefacts and their paths.
- **Decisions made:** choices and their supporting evidence.
- **Unresolved issues:** remaining uncertainty and risks.
- **Next recommended action:** one concrete next action, not a broad list of ideas.

Do not overwrite raw data, existing experiment results or the user's source files without explicit instruction. Prefer additive changes and small, reviewable commits.

---

## 1. Project objective

### 1.1 Central objective

Build and evaluate a reproducible financial time-series classification pipeline under controlled, structured corruption of its input data. Apply the same experimental protocol independently to a US equity panel and a Norwegian equity panel. Use a small set of course-relevant classifiers and a deterministic C++17 fault-injection component integrated with Python.

The primary scientific question is:

> **How do controlled forms of financial time-series data corruption propagate through feature construction and affect volatility-transition classification, and to what extent can one carefully selected mitigation strategy reduce the resulting degradation across two independently sourced equity datasets?**

The primary engineering question is:

> **Can the fault injection, data processing and evaluation process be made deterministic, observable, testable and reproducible, so that a change in model performance can be traced to the fault mechanism that caused it?**

The project is not primarily an exercise in finding the highest clean-data accuracy. It is a controlled study of reliability under violated input assumptions.

### 1.2 Research subquestions

1. **Data provenance and quality:** What are the structural and economic properties of each equity panel and its benchmark series? What is known and not known about price adjustments, coverage, missingness and instrument membership?
2. **Fault propagation:** How do missing observations, stale prices, erroneous price spikes and temporal benchmark misalignment affect returns, rolling realised volatility and derived features?
3. **Classifier robustness:** How does predictive performance change with fault type and severity for logistic regression, random forest and a feed-forward neural network?
4. **Mitigation:** Does one selected mitigation strategy reduce corruption-induced degradation without an unacceptable loss of clean-data performance?
5. **Cross-market replication:** Are the *within-market changes in performance* similar when the same protocol is applied independently to the US and Oslo datasets?
6. **Engineering verification:** Can the C++ fault injector produce controlled, reproducible modifications and event logs that agree with the actual modified data?

### 1.3 What would count as a useful result?

Any of the following can be a valid result, provided the experiment is correctly designed and interpreted:

- one model is consistently more robust than another for a specified fault family;
- a particular fault propagates through rolling features in a predictable way;
- a mitigation improves corrupted-input performance, with or without a clean-performance trade-off;
- a mitigation fails to help, under a sufficiently informative experiment;
- a fault detector identifies injected defects but also generates unacceptable false alarms on genuine market events;
- the robustness pattern differs between the US and Oslo panels, with plausible data- or market-specific explanations;
- the data are insufficient to support a proposed target or comparison, and the scope is responsibly narrowed.

A negative result is not automatically evidence that transitions are intrinsically unpredictable. It may reflect a weak feature, noisy labels, low event counts, insufficient sample size, poor calibration, inappropriate fault severity or an underpowered test.

---

## 2. Agreed scope and boundaries

| Component | Current decision | Status |
|---|---|---|
| US equity data | Kaggle S&P 500 individual-stock price panel | Core |
| US benchmark | SPY, acquired separately from the Kaggle panel | Core |
| Norwegian equity data | Existing Oslo adjusted-price workbook | Core, subject to target feasibility and source audit |
| Norwegian benchmark | OSEBX, provisionally `OSEBX.OL`; inspect actual returned fields and coverage | Core, subject to verification |
| Dataset comparison | Run the same protocol independently on each market; compare within-market degradation, not unqualified raw scores | Core |
| Learning task | Individual-stock calm-to-high volatility transition classification | Provisional target; must pass feasibility gate |
| Classifiers | Logistic regression, random forest, feed-forward neural network | Core, satisfies Path B method requirements if implemented/evaluated properly |
| Fault types | Missing values/blocks, stale-price blocks, bad-tick price spikes; benchmark misalignment as a separate scenario | Core/benchmark scenario subject to implementation time |
| Native code | C++17 fault injector; Eigen; pybind11 if build environment allows | Core engineering deliverable |
| Main ML implementation | Python and established libraries or the student's verified Project 2 implementation | Core |
| Cross-sectional volatility dispersion | Optional feature ablation after baseline | Secondary |
| PCA/SVD, spectral and wavelet features | Deferred | Not required for MVP |
| Sector ETF architecture from earlier work | Do not reproduce | Excluded |
| C++ neural-network training | Do not implement | Excluded |
| Agent orchestration / synthetic users | Not part of this financial experiment | Excluded |
| Lawyer allocation and retention modelling | Not addressed by this dataset | Excluded |

### Important interpretation of the two datasets

- **US experiment:** Kaggle individual equities + SPY benchmark.
- **Norwegian experiment:** Oslo individual equities + OSEBX benchmark.
- Do not concatenate both markets into a single undifferentiated panel.
- Do not assume that the two providers use identical adjustment methodology.
- Do not call the Oslo data “perfect” or a gold standard. A structural audit is not proof of economic correctness.
- Do not compare raw model scores across markets as if market, data vendor, universe, date range and sample size were controlled. The more defensible comparison is the degradation relative to each market's own clean baseline.
- Do not assume that the Kaggle panel is a point-in-time historical S&P 500 constituent universe. Investigate constituent history, delistings and survivorship bias before making such a claim.
- Do not use XLK, XLF, XLV or a sector-ETF panel. The project does not need them.

---

## 3. Data sources and current implementation state

### 3.1 US equity panel: Kaggle

Dataset handle currently used:

`darkmatternet/s-and-p-500-stocks-25-years-of-data-updated-daily`

Existing module:

`Kaggle_Data_download.py`

Current functionality:

- downloads/caches the Kaggle dataset through `kagglehub`;
- searches recursively for CSV files;
- identifies date, price and ticker columns using candidate names;
- converts CSVs into a long `[date, ticker, adj_close]` representation;
- pivots to a wide date-by-ticker DataFrame.

A previous terminal run reported a panel of shape `(6762, 503)`. Treat this as a historical observation from that particular run, not a fixed specification. Re-run the loader and save the actual dimensions, date range, columns, file names and snapshot hashes.

**Known loader issue:** the current `_load_long_csv()` allows `close` as a fallback when an adjusted-close column is unavailable, while naming the output `adj_close`. This can silently mix adjusted and unadjusted series. Change the contract: either require adjusted close, or explicitly return a differently named unadjusted-close field and record which price definition is being used. Never silently mix the two.

SPY was previously absent from the Kaggle columns. This is expected: the downloader contains individual equities, not necessarily the ETF benchmark. Acquire SPY separately and join by date after inspecting the returned schema.

### 3.2 US benchmark: SPY

- Provisional source: Yahoo Finance via `yfinance`.
- Ticker: `SPY`.
- Explicitly set price-adjustment options rather than relying on package defaults.
- Inspect returned columns. If using `auto_adjust=False`, check whether `Adj Close` is present; do not assume the schema without printing it.
- Verify date range, missing sessions, duplicate dates, currency/price semantics and the overlap with the Kaggle panel.
- Report how many dates are lost under any join operation. Do not let an `inner` join silently define the analytical sample.

### 3.3 Norwegian equity panel: Oslo workbook

Existing module:

`data/oslo_prices.py`

Existing functions:

- `load_oslo_prices(...)`
- `audit_price_panel(prices)`

The loader reads the `Adjusted Prices` sheet using `header=2`, parses `Date`, sorts the index, converts price columns to numeric and rejects unexpected non-numeric values. The audit reports observations, instruments, date range, missing cells, infinite values, non-positive prices, duplicate dates and missingness by instrument.

**Important audit interpretation:** `no_missing_price_cells` is a strict structural check, not a universal definition of data quality. A stock may legitimately lack prices before listing or after it stops trading. Separate expected coverage boundaries from unexplained internal gaps. A missing cell must be classified, not automatically treated as a bad source record.

Re-run the audit on the exact workbook to be used. Do not rely on an old description that the panel contains five stocks or 751 dates until the current workbook has been inspected.

### 3.4 Norwegian benchmark: OSEBX

- Provisional ticker: `OSEBX.OL` through Yahoo Finance.
- Verify that the symbol downloads and inspect the exact columns and price semantics.
- An index series may expose `Close` rather than `Adj Close`. If so, name and document it as index close; do not relabel it adjusted close.
- Verify the overlap with Oslo stock dates, exchange holidays and missing sessions.
- If a trustworthy benchmark series is not available or the fields are not comparable, document the limitation. Do not fabricate a benchmark or silently substitute a different index.

### 3.5 Data-provenance record

For every source and frozen snapshot, record:

- provider and dataset identifier;
- source URL or Kaggle handle;
- retrieval timestamp;
- exact filenames and SHA-256 hashes;
- price field and adjustment semantics;
- date range and number of observations;
- instrument universe and inclusion rule;
- package versions and acquisition code commit;
- missingness and coverage diagnostics;
- known survivorship, corporate-action, currency or vendor limitations.

The Kaggle dataset may be updated. A reproducible experiment must use a retained snapshot, not a live download that changes between runs.

---

## 4. Observation unit, time-series construction and target

### 4.1 Observation unit

The planned supervised-learning sample is an **instrument-date pair** `(instrument i, prediction date t)`. All stocks observed on a given date must remain in the same train, validation or test partition. Benchmark and cross-sectional features are date-level quantities and may be repeated across instruments for that date.

### 4.2 Prediction-time convention

Provisional convention: prediction is made **after the close of session t**, once the adjusted closing prices for session `t` are available. Consequently, features may use observations through `t`, but no observation after `t`.

If a different prediction-time convention is chosen, adjust every feature and alignment rule consistently and record the change.

### 4.3 Returns

For positive adjusted closing price `P[i,t]`:

\[
r_{i,t}=\log(P_{i,t})-\log(P_{i,t-1})
      =\log\left(P_{i,t}/P_{i,t-1}\right).
\]

Use a consistent price definition per series. Do not forward-fill a missing clean price before calculating the clean reference return. The difference between two observed rows is not automatically a one-session return if intervening sessions are missing.

### 4.4 Realised volatility

For a trailing window of `h` sessions:

\[
RV_{i,t}^{(h)} = \sqrt{\sum_{j=0}^{h-1} r_{i,t-j}^{2}}.
\]

For the following `h` sessions:

\[
RV_{i,t}^{+, (h)} = \sqrt{\sum_{j=1}^{h} r_{i,t+j}^{2}}.
\]

The provisional primary horizon is `h = 5`. This is the square root of the sum of squared returns (not the root mean square and not annualised). Use the same convention for each asset and benchmark. If a different scaling is chosen, name it and apply it consistently.

A rolling implementation must preserve date alignment. For example, the trailing measure at `t` uses returns `t-4, ..., t`; the future measure aligned to `t` uses returns `t+1, ..., t+5`.

### 4.5 Provisional target: calm-to-high transition

For each instrument, calculate a training-only threshold:

\[
q_{i,0.80}^{train}=Q_{0.80}\left(\{RV_{i,t}^{(5)}:t\in\mathcal{T}_{train}\}\right).
\]

Define current and forward high-volatility states:

\[
S_{i,t}=\mathbb{1}[RV_{i,t}^{(5)}>q_{i,0.80}^{train}],
\qquad
S_{i,t}^{+}=\mathbb{1}[RV_{i,t}^{+,(5)}>q_{i,0.80}^{train}].
\]

The candidate binary label is:

\[
y_{i,t}=\mathbb{1}\left[
RV_{i,t}^{(5)}\le q_{i,0.80}^{train}
\;\land\;
RV_{i,t}^{+,(5)}>q_{i,0.80}^{train}
\right].
\]

Interpretation:

- `1`: the stock is not currently in the high-volatility state and enters the high-volatility state over the next five sessions;
- `0`: all other cases under this binary definition.

This is specifically a **calm-to-high early-warning target**, not a label for every possible state transition. Retain the four-way transition counts (calm-to-calm, calm-to-high, high-to-calm, high-to-high) as diagnostics.

The threshold is estimated on training data only and frozen. Do not tune the threshold using validation/test outcomes in a way that optimizes reported test performance. The future volatility variable is target-construction-only and must never enter the feature matrix.

### 4.6 Target feasibility gate

Before model training, report for each market, partition and (where useful) instrument:

- number of valid instrument-date samples;
- positive and negative counts;
- positive-class fraction;
- number of distinct dates containing positives;
- number of contiguous event blocks containing positives;
- per-instrument target counts;
- number of instruments with adequate observations.

If Oslo has too few positive events or too few distinct positive-event blocks, do not force the same target merely for symmetry. The valid options are:

1. retain Oslo as a descriptive/data-quality replication while the US panel is the primary classifier experiment;
2. use a predeclared alternative target/horizon justified by training/validation evidence;
3. report that the two-market classification comparison is infeasible with the available Oslo sample.

Do not select whichever threshold produces the most attractive test result.

---

## 5. Feature design

Start with a small, interpretable feature set. All transformations must be causal with respect to the prediction time and fitted using training data only where fitting is required.

### 5.1 Core asset-level features

Candidate initial features:

- latest return `r[i,t]`;
- absolute return `abs(r[i,t])`;
- trailing realised volatility `RV[i,t](5)`;
- trailing realised volatility `RV[i,t](20)`;
- short/medium volatility ratio `RV5 / (RV20 + epsilon)`.

Add only if justified and feasible: rolling robust return dispersion, drawdown, rolling skewness/kurtosis or other time-domain measures. Do not add a large feature zoo before the baseline is working.

### 5.2 Benchmark-context features

For each market's own benchmark:

- latest benchmark return;
- five-session benchmark realised volatility;
- twenty-session benchmark realised volatility.

US uses SPY; Oslo uses OSEBX if verified. Compare asset-only features with asset-plus-benchmark features so the benchmark's incremental contribution is measurable.

### 5.3 Optional cross-sectional dispersion

The proposed feature is **cross-sectional dispersion of realised volatility**, not “cross-sectional vol-of-vol”:

\[
D_t=\operatorname{Std}_{i\in\mathcal{U}_t}(RV_{i,t}^{(5)}),
\]

where `U_t` is the explicitly defined set of eligible instruments with valid volatility values on date `t`.

Potential interpretation: does the distribution of current volatility across assets provide information about the probability of an individual stock's future transition?

Constraints:

- It is identical for all instruments at a given date; it represents market/cross-sectional context, not a stock-specific feature.
- Record the number of valid instruments `N_t` used on each date.
- Changing universe composition can move `D_t` even when each stock's own volatility is unchanged.
- Compare with cross-sectional mean volatility; dispersion may merely proxy for overall market volatility.
- For small Oslo universes, treat it as exploratory and do not assume it is reliably estimated.
- Run an ablation with and without it. Do not claim novelty without a literature review.

A relative-volatility-anomaly dispersion measure may be considered later, but it is not part of the minimum feature set.

### 5.4 Data-quality indicators

Possible observable indicators include missingness flags, repeated-price run lengths, return discontinuity flags and robust outlier flags. The model must not receive the true synthetic injection mask. The injection mask is oracle information for evaluating the fault injector or a quality detector only.

Keep “market extremeness” and “suspected data corruption” as distinct concepts. A large return is not automatically a bad tick.

### 5.5 Deferred features

PCA/SVD, spectral entropy, wavelet energy and high-dimensional cross-asset features are deferred until the minimum experiment is complete. Add them only when a specific hypothesis, test and time budget justify the extra complexity.

---

## 6. Chronological splitting and leakage control

Use chronological partitions separately for each market. A provisional starting point is 70% train, 15% validation and 15% test by unique trading dates, but adjust only on the basis of documented coverage and event counts. Store exact date boundaries in a split manifest.

Mandatory rules:

1. Split by unique date, not by shuffled instrument-date rows. All instruments on the same date belong to the same partition.
2. Purge training/validation/test boundary samples whose forward target window crosses into the next partition. For a five-session forward target, at least the relevant five sessions must be considered; verify actual target-interval overlap instead of relying on a row-count assumption.
3. The final test labels require five future sessions of clean outcomes. Either reserve an outcome-only buffer beyond the feature test interval or exclude the final five prediction dates from scoring. Do not use those future observations as test-time features.
4. Historical feature windows at the start of validation/test may use past data from the preceding period because that information would have been available at the prediction time.
5. Fit imputers, scalers, feature selectors, thresholds and calibrators using training data only (or the relevant training fold).
6. Perform hyperparameter selection on training/validation data. Test data are used once after the pipeline is frozen.
7. Keep data snapshot, date split, feature version, label version and random seeds fixed across paired clean/corrupted comparisons.

Use a walk-forward or expanding-window validation approach within the training period if feasible. Record any deviation.

---

## 7. Fault-injection design

### 7.1 Principle

Maintain two distinct data objects:

- **Clean reference:** immutable prices from which the correct labels and clean reference features are derived.
- **Corrupted input copy:** an experimental copy to which faults are applied before the feature pipeline, representing what the model would receive from a faulty upstream data source.

For the primary robustness evaluation, the classifier is evaluated on features derived from the corrupted copy but scored against labels derived from the clean reference. This isolates the effect of input corruption on predictions.

The injected faults must not change the ground-truth labels. Do not feed the injection mask or original clean values to the classifier.

Apply test-time faults only to the held-out feature inputs in the clean-trained robustness experiment. Ensure rolling-window propagation and any context needed at test start are handled explicitly. Faults should not leak into training or validation unless the separate augmentation experiment deliberately injects them there.

### 7.2 Core fault families

| Fault | Controlled operation | Key parameters to record | Expected propagation to inspect |
|---|---|---|---|
| Random missingness | Set selected asset-price cells to missing | affected fraction, selected assets/dates, seed | invalid return and rolling windows; imputation and coverage behaviour |
| Contiguous missing block | Remove consecutive observations for selected assets | block length, start date, assets, seed | rolling feature gaps and sample-availability loss |
| Stale-price block | Repeat a previous price for a specified run | block length, affected assets, restoration rule | artificially suppressed returns followed by a catch-up movement |
| Bad tick / price spike | Apply a one-observation multiplicative/log-price error and restore the correct price next session | affected asset/date, magnitude, sign, seed | two artificial return effects and rolling-volatility contamination |
| Benchmark misalignment | Shift benchmark observations/features by one session | direction and lag length | false market context and misattribution of shared movements |

The first three asset-level corruption families are the minimum implementation. Benchmark misalignment can be a separate Python alignment test if adding it to the C++ injector would delay the core deliverable.

Severity levels must be defined before final test evaluation. Use interpretable levels and retain exact parameters. A pilot can determine whether candidate severities are too weak or trivially destructive, but freeze the evaluation grid before looking at final test metrics.

### 7.3 Negative controls and invariance tests

At minimum include:

- **No-fault identity:** injecting zero faults leaves the matrix unchanged.
- **Global multiplicative scale invariance:** for a complete price series, replacing `P[i,t]` by `c * P[i,t]` for a positive constant `c` leaves log returns and realised volatility unchanged, up to floating-point tolerance.
- **Genuine market shock control:** include at least one documented high-volatility market episode without injecting faults and check that a quality detector or mitigation does not automatically classify every extreme move as corrupt.
- **Fault locality:** only the intended asset/date cells or block may be directly modified; downstream feature effects must be traceable and expected.

Do not use a fixed calendar event as evidence of a genuine shock without verifying its source and dates. Event annotation is descriptive, not ground truth for model labels.

### 7.4 Experimental conditions

**Condition A — Clean baseline**

- Train and select the model using clean training/validation inputs.
- Evaluate on clean test inputs.
- Retain predictions, probabilities and all metrics.

**Condition B — Corrupted test input**

- Keep the trained model, fitted preprocessing, split and labels fixed.
- Corrupt a copy of the held-out test input data.
- Re-run the ordinary feature-processing pipeline.
- Evaluate against the original clean labels.
- Measure degradation relative to Condition A.

**Condition C — One mitigation experiment**

First establish that the clean-trained model is affected by the chosen faults. Then select one mitigation strategy for the MVP:

Preferred candidate: **fault-augmented training**, where additional corrupted versions of training inputs are added while labels remain the clean labels. Evaluate on unseen injection seeds/patterns and the same clean reference test labels.

Do not combine augmentation, quality-indicator features, imputation changes and multiple detector strategies into one condition. That would make it impossible to identify which mechanism caused an improvement. Treat any additional mitigation as a separate ablation.

### 7.5 Fixed-cohort and coverage rule

Corruption may make some feature rows unavailable. Do not silently drop difficult test rows and report metrics only on the survivors.

Report both:

- predictive metrics on the evaluable sample cohort;
- **coverage/availability:** fraction of originally eligible test samples for which a prediction was produced.

Where possible, use an explicit, training-fitted imputation policy and missingness indicators. If a pipeline abstains or cannot produce a prediction, count and report those cases separately. Compare clean and corrupted metrics on matched sample identifiers.

---

## 8. C++17 component: scope and interface

### 8.1 Responsibility

The C++ component is a deterministic, testable **fault-injection engine**, not a second machine-learning stack.

Python remains responsible for:

- data acquisition and pandas-based panel management;
- clean data validation and feature/target construction;
- model training and hyperparameter selection;
- statistical analysis, metrics and plots;
- experiment orchestration and report artefacts.

C++ is responsible for:

- applying specified controlled perturbations to a numeric price matrix or a precomputed fault schedule;
- returning the corrupted matrix;
- returning structured event records describing exactly what changed;
- supporting deterministic execution and unit tests.

### 8.2 Proposed implementation

- C++17.
- Eigen for dense numerical matrices (`T x N`, rows = dates, columns = instruments).
- `std::mt19937_64` or an explicitly supplied event schedule for reproducible selection.
- pybind11 for Python/NumPy integration if the local compiler/build environment supports it.
- If pybind11 setup delays the project, first build and test a standalone C++ executable or library. Do not let binding infrastructure block validation of the injection logic.

### 8.3 Conceptual interface

Input:

- price matrix;
- ordered date metadata and instrument identifiers (or a stable mapping maintained by Python);
- fault configuration: type, severity, affected fraction, block length and seed.

Output:

- corrupted matrix (without mutating the clean input object);
- event log with event ID, fault type, instrument, date/interval, original value, modified value and configuration;
- run metadata and seed.

The exact implementation may use a precomputed event schedule supplied by Python if that makes the C++ logic simpler and more reproducible. Persist the realised event schedule, not only the random seed. Cross-platform bitwise equality of standard-library random distributions should not be assumed without verification.

### 8.4 Required tests

1. no-fault identity;
2. same input/configuration/seed gives the same result on the supported build;
3. only intended cells/intervals change;
4. event log reconciles with every modification;
5. requested fault magnitude/severity is respected;
6. invalid indices, non-positive input values and malformed configurations are handled explicitly;
7. input matrix is not unexpectedly mutated;
8. missing values are represented consistently (for example IEEE NaN for numeric matrices);
9. C++ result agrees with a simple Python reference on fixed deterministic fixtures within a stated tolerance;
10. benchmark/date-shift behaviour is checked separately if implemented outside the core injector.

### 8.5 Build strategy and risk

First verify a working compiler on Windows (`cl`, `g++` or `clang++`) and CMake. Then build the smallest C++17 program. Add Eigen next. Add pybind11 last. Do not introduce LibTorch or write an MLP in C++ for this project.

The user previously encountered a Python 3.15 / PyYAML build failure while installing `kagglehub`, where pip attempted a source build and reported missing Microsoft Visual C++ Build Tools. Treat this as a known environment risk. Inspect the current interpreter and packages; use a dedicated virtual environment with a currently supported scientific-Python version if needed. Do not assume that `pip` and `python` refer to the same interpreter.

No speedup claim is permitted without a fair benchmark against an equivalent NumPy/reference implementation using the same fault semantics, input sizes, warm-up and repeated measurements.

---

## 9. Models and learning protocol

### 9.1 Required model set

1. **Logistic regression:** linear baseline. Scale continuous features using parameters fitted on training data only.
2. **Random forest:** tree-based ensemble. Tune a compact, justified grid for number of trees, depth, minimum leaf size and feature subsampling.
3. **Feed-forward neural network (MLP):** one or two hidden layers. Reuse a verified Project 2 implementation or use a documented library implementation. Tune a compact grid for width, learning rate, regularisation and early stopping.

All models must use the same eligible sample IDs, target definition, time split and core feature set for a fair comparison. If a model cannot handle missing data directly, the imputation pipeline must be fitted on training data only and applied consistently.

### 9.2 Baselines

- **Always-negative baseline:** predicts no calm-to-high transition. This exposes the consequence of class imbalance and must be reported.
- Do not call the previous-label rule a persistence baseline unless its semantics are explicitly defined for this particular transition target.
- A simple volatility-threshold or other rule-based baseline may be added if it is defined before test evaluation and does not leak future information.

### 9.3 Hyperparameter selection

- Use a validation split or time-aware resampling within the training period.
- Fit preprocessing, scaling, imputation and calibration on training data only (or within each training fold).
- Select classification thresholds on validation data and freeze them before test evaluation.
- Do not repeatedly inspect test performance while developing the model.
- Record hyperparameters, seeds, training time, inference time, model size/parameter count where applicable, convergence status and package versions.
- Present hyperparameter search as curves or heatmaps when useful, not as an unstructured list of trial runs.

### 9.4 Feature ablation order

1. asset-level time-domain features;
2. asset + benchmark context;
3. asset + benchmark + cross-sectional dispersion, only if the cross-section is sufficiently complete;
4. one quality-aware or fault-augmentation mitigation condition.

This ordering makes it possible to attribute changes to information added rather than to model complexity.

---

## 10. Evaluation and statistical analysis

### 10.1 Course-required metrics

Report on the same held-out test observations for the three models:

- accuracy and its standard error;
- confusion matrix;
- per-class precision, recall and F1;
- macro-F1;
- balanced accuracy;
- ROC curve and ROC-AUC for the binary task (one-vs-rest only if the task is later changed to multiclass).

### 10.2 Robustness-specific metrics

- average precision / PR-AUC for the transition class;
- Brier score and/or log loss for predicted probability quality;
- transition recall at a validation-selected operating threshold;
- false alarms per year or per fixed number of eligible trading days;
- performance degradation `M_corrupted - M_clean` for each metric;
- robustness curve versus fault severity;
- coverage / fraction of samples for which a prediction was produced;
- fault-detector precision/recall/FPR if a separate observable quality detector is implemented.

Accuracy alone is not sufficient, especially if the positive transition class is rare.

### 10.3 Paired comparisons and uncertainty

The panel is dependent in several ways:

- successive five-session forward targets overlap;
- rolling features overlap in time;
- multiple stocks share the same market and benchmark conditions;
- each instrument produces repeated observations.

Do not treat every instrument-date row as an independent replicate.

Provisional approach:

1. Pair clean and corrupted evaluations on the same test dates and sample IDs.
2. Keep all instruments observed on a date together when resampling.
3. Use a moving-block or stationary-block bootstrap over contiguous trading dates.
4. A 20-session starting block length is a candidate because the core features include a 20-session volatility window; inspect dependence and test sensitivity to alternative lengths such as 10 and 40 sessions before freezing the final choice.
5. Use at least 1,000 bootstrap replicates if feasible; record method and seed.
6. Report the number of distinct positive-event blocks. If too few exist, state that robust uncertainty quantification is not possible.
7. Report the course-requested accuracy standard error, but explain that the conventional independent-observation formula can understate uncertainty for dependent panel data.

A mixed-effects or factorial model of iteration-level performance may be explored only if the design supplies enough independent blocks/replicates. Start with paired differences, plots and block-bootstrap intervals. Do not fit a sophisticated model merely because many rows exist.

### 10.4 Iteration-level variables

Potential explanatory/design variables:

- dataset and benchmark;
- data snapshot hash;
- model family and hyperparameters;
- feature-set version;
- fault family and severity;
- affected fraction / block length / magnitude;
- fault seed and model seed;
- training strategy (clean or augmented);
- code commit and environment version.

Potential response variables:

- PR-AUC, ROC-AUC, macro-F1, balanced accuracy;
- transition recall and false alarms;
- Brier score/log loss;
- coverage;
- data-quality detector metrics;
- training/inference runtime;
- C++ injection runtime, only if benchmarked fairly.

The iteration is a recorded experiment configuration. It is not automatically an independent statistical observation.

---

## 11. Traces, observability and mechanism-level diagnosis

### 11.1 Distinguish output change from mechanism attribution

An output-level statement is: “macro-F1 fell by 0.12 under a price-spike scenario.”

A mechanism-level explanation is: “the injected bad tick altered price `P[i,t]`; this created a positive return at `t` and a reversal return at `t+1`; both altered squared returns and the next five trailing volatility windows; the robust outlier check fired/did not fire; the model's probability changed by a measured amount for the affected sample IDs.”

The second statement is the engineering objective. Retain enough lineage to test it.

### 11.2 Minimum experiment manifest

Every meaningful run should save JSON or YAML with:

- `run_id`, timestamp, elapsed time;
- `dataset_id`, benchmark ID, raw snapshot hashes;
- code commit and dirty-tree flag;
- Python, package, compiler, Eigen and pybind11 versions;
- split ID and exact train/validation/test dates;
- feature version and ordered feature names;
- label version, horizon, threshold method and threshold values;
- sample counts, class counts and coverage;
- model type, hyperparameters, seed and fit time;
- fault type, severity, affected fraction, block length, seed and realised event schedule;
- quality-check outputs and counts;
- metrics, decision threshold and uncertainty intervals;
- artifact paths for predictions, probabilities, plots, tables, logs and model files.

### 11.3 Fault-event log

For each injected event, retain at least:

- event ID;
- fault family;
- instrument identifier;
- date or interval;
- original value(s);
- modified value(s);
- injection parameters and seed;
- affected cell coordinates;
- corresponding sample IDs and feature windows affected, where computable.

### 11.4 Minimum mechanism trace

For selected illustrative events, trace:

`fault injection -> modified price -> affected return(s) -> affected rolling feature(s) -> quality-check result -> model probability/class -> evaluation outcome`.

Retain clean and corrupted values side by side. Do not include secret credentials, API keys or private account data in manifests or repositories.

---

## 12. Software structure and reusable documents

A proposed layout (adapt to the repository after inspection; do not create duplicate or conflicting modules):

```text
Project3/
├── data/
│   ├── raw/                 # immutable local snapshots; do not commit large/private files blindly
│   ├── processed/           # derived clean panels/features
│   └── manifests/           # snapshot, provenance and split metadata
├── cpp/
│   └── fault_injector/
│       ├── include/
│       ├── src/
│       └── tests/
├── python/
│   ├── acquisition/
│   ├── validation/
│   ├── features/
│   ├── labels/
│   ├── models/
│   ├── experiments/
│   └── evaluation/
├── tests/
├── results/
│   ├── metrics/
│   ├── predictions/
│   ├── figures/
│   └── fault_events/
├── docs/
│   ├── PROJECT_SPEC.md
│   ├── DATA_CARD_US.md
│   ├── DATA_CARD_OSLO.md
│   ├── FAULT_TAXONOMY.md
│   ├── EXPERIMENT_PROTOCOL.md
│   ├── TRACE_SCHEMA.md
│   ├── DECISIONS.md
│   └── STATE_OF_WORK.md
├── requirements.txt
└── README.md
```

### Required maintained documents

- **PROJECT_SPEC.md:** problem, intended purpose, scope, assumptions, constraints, non-goals, acceptance criteria.
- **DATA_CARD_US.md / DATA_CARD_OSLO.md:** origin, price semantics, coverage, universe, missingness, limitations and hashes.
- **FAULT_TAXONOMY.md:** fault mechanisms, parameterisation, severity, expected propagation, realism limits.
- **EXPERIMENT_PROTOCOL.md:** hypotheses, target, split, models, metrics, repetitions, bootstrap, decision thresholds.
- **TRACE_SCHEMA.md:** machine-readable fields and artifact naming/retention rules.
- **DECISIONS.md:** date, decision, alternatives, evidence, rationale and consequences.
- **STATE_OF_WORK.md:** what exists, what was run, what passed/failed, unresolved uncertainty and next action.

These documents exist to preserve product and development context across agents and separate coding sessions. They must be updated from observed evidence, not filled with speculative progress.

---

## 13. Workflow test panel

Run the relevant tests after every meaningful change. The list below is conceptual; implement tests incrementally.

| Workflow | Failure/suboptimal behaviour | Test/evidence |
|---|---|---|
| Acquisition | wrong schema, missing ticker, wrong price field, incomplete download, changing snapshot | schema contract, file inventory, row/ticker/date counts, hashes, source/version record |
| Price semantics | adjusted and unadjusted prices silently mixed | require named field; provenance metadata; explicit rejection or documented alternative |
| Date alignment | benchmark shifted one session; holidays mishandled; duplicate dates; hidden join losses | date-index checks, overlap report, one-session-shift test, join row-count audit |
| Clean panel | unexplained gaps, non-positive/infinite prices, changing universe | structural audit, per-instrument coverage, internal gap report, universe counts |
| Returns | wrong ordering, wrong denominator, false one-session return across gap | hand-calculated fixtures, date continuity test, zero/scale invariance tests |
| Rolling features | off-by-one window, future data included, incorrect `shift` | hand-worked time-index fixtures, feature timestamp assertions, window-boundary tests |
| Labels | future information in features; threshold fitted on test; target window crosses split | timestamp audit, training-only threshold provenance, boundary purge assertions, hand-labelled fixtures |
| Fault injector | non-determinism, wrong severity, unintended cells, bad event log | fixed-seed test, locality test, event reconciliation, no-op identity, invalid-config tests |
| Preprocessing | scaler/imputer fitted on test; missing rows silently dropped | fit-transform provenance, train-only assertions, sample/coverage accounting |
| Model training | inconsistent feature order, seed drift, class collapse, non-convergence | saved feature schema, seeds, convergence log, probability distribution and class-count checks |
| Evaluation | test used for tuning; rows differ between comparisons; dependent rows treated as IID | frozen split/model/threshold, matched IDs, paired analysis, block bootstrap |
| Reporting | figure/table stale or does not match metrics | generate outputs from run registry; clean rerun from manifest; commit/environment recorded |
| C++/Python bridge | memory/order/dtype mismatch, accidental mutation | fixed matrix fixtures, shape/dtype/index checks, input-copy test, parity tolerance |

### Minimum regression tests before any modelling result is trusted

- `prices` are positive and finite where present;
- dates are sorted and unique;
- returns are calculated only across valid consecutive observations under a documented calendar rule;
- trailing volatility at `t` uses only returns through `t`;
- forward volatility uses exactly `t+1` through `t+h`;
- target thresholds are training-only;
- no future-derived field is in `X`;
- all stocks for one date stay in one split;
- purged target intervals do not cross partitions;
- clean labels are identical across all fault conditions;
- no-fault injection returns the original values;
- event log equals the set of actually modified cells;
- model test comparisons use matched sample IDs and the same target labels.

---

## 14. Proposed one-week MVP sprint

The one-week sprint is a **prototype milestone**, not the complete course submission. The course deadline remains 11 December 2026.

### Day 1 — Data inventory, acquisition and provenance

- [ ] Inspect repository and current Python environment.
- [ ] Load Kaggle panel and print actual shape, columns, date range and CSV fields.
- [ ] Freeze a local Kaggle snapshot and hash source files.
- [ ] Correct the adjusted-close fallback issue.
- [ ] Acquire SPY separately and inspect schema/coverage.
- [ ] Load and audit Oslo workbook.
- [ ] Acquire/inspect OSEBX; verify ticker, field and date overlap.
- [ ] Produce one data card per market.

**Acceptance:** the two panels and their benchmarks load reproducibly; source fields and overlap are documented; no silent price-field substitution occurs.

### Day 2 — Time-series construction and target feasibility

- [ ] Build clean stock and benchmark return series.
- [ ] Build five- and twenty-session trailing realised volatility.
- [ ] Build forward five-session target quantities from clean data.
- [ ] Implement the provisional training-only 80th-percentile calm-to-high label.
- [ ] Report sample counts, class balance and distinct event blocks by market/partition.
- [ ] Create chronological partitions and purge target-window overlaps.
- [ ] Plot representative prices, returns, realised volatility, target-state timeline and class counts.

**Acceptance:** clean features/labels pass hand-calculation and timestamp tests; target feasibility is explicitly assessed. Narrow the Oslo classification scope if event counts are inadequate.

### Day 3 — C++17 fault injector

- [ ] Verify compiler and CMake availability.
- [ ] Build minimal C++17 project.
- [ ] Implement no-op and one deterministic fault first.
- [ ] Add missing block, stale-price block and price spike.
- [ ] Add structured event log.
- [ ] Add unit tests for determinism, locality, parameter compliance and event reconciliation.
- [ ] Add pybind11 integration only if it does not endanger the core build.

**Acceptance:** component builds, tests pass and one small Python integration test can invoke it. If binding is blocked, preserve a working standalone component and report the limitation.

### Day 4 — Clean baseline models

- [ ] Implement always-negative baseline.
- [ ] Train logistic regression, random forest and MLP.
- [ ] Tune compact hyperparameter grids using training/validation only.
- [ ] Freeze model and preprocessing configurations.
- [ ] Evaluate once on clean test data.
- [ ] Save predictions/probabilities and run manifests.

**Acceptance:** all models use matched samples, feature order, target and splits; course metrics and robustness-relevant baseline metrics are saved.

### Day 5 — Fault robustness experiment

- [ ] Corrupt copies of held-out input series, not the clean reference labels.
- [ ] Test missingness, stale blocks and price spikes at predeclared severities.
- [ ] Recompute features through the ordinary pipeline.
- [ ] Track sample coverage and all fault events.
- [ ] Evaluate each fixed baseline model against clean labels.
- [ ] Plot performance against fault type/severity.

**Acceptance:** at least two fault families produce deterministic, traceable, interpretable experiments with matched clean/corrupted sample accounting.

### Day 6 — One mitigation and statistical analysis

- [ ] Choose one mitigation after inspecting clean-trained degradation.
- [ ] Preferred MVP candidate: fault-augmented training with clean labels and unseen test fault seeds/patterns.
- [ ] Compare clean and corrupted performance before/after mitigation.
- [ ] Calculate paired degradation and uncertainty using date blocks where feasible.
- [ ] Compare the US and Oslo within-market degradation patterns if both datasets pass feasibility.

**Acceptance:** mitigation is evaluated without confounding multiple simultaneous changes; clean-data cost and corrupted-data benefit are both reported.

### Day 7 — Documentation, reproducibility and review

- [ ] Finalise PROJECT_SPEC, data cards, fault taxonomy and experiment protocol.
- [ ] Finalise experiment manifest and fault event schema.
- [ ] Generate figures/tables from saved run artefacts.
- [ ] Verify a fresh rerun from documented environment and commands.
- [ ] Update README and STATE_OF_WORK.
- [ ] Record remaining limitations and the next experiment.

**Acceptance:** another developer can identify the exact data snapshot, reconstruct the experiment configuration, run the pipeline and understand which claims are supported by results.

---

## 15. Internship work-package alignment

| Internship work package | Transferable artefacts from this project | Boundary |
|---|---|---|
| WP1 — traces, observability, tests, evaluations | fault taxonomy, event logs, run manifests, regression tests, metrics, fault-to-feature-to-prediction traces | Directly demonstrated in a financial testbed |
| WP2 — documents and context-rot prevention | project specification, data cards, experiment protocol, trace schema, decision log, state-of-work record | Demonstrates documentation practice; does not build the agentic development platform |
| WP3 — statistical analysis | repeated experiment table, paired comparisons, block bootstrap, uncertainty and performance-degradation analysis | Inference remains bounded by financial panel dependence and sample size |
| WP4 — ideation | combined faults, false-positive controls, mitigation hypotheses, fault propagation graphs | Ideas remain hypotheses until tested |
| WP5 — experiment execution | clean/corrupted/mitigated comparisons across models and datasets | One bounded experiment, not the full internship experiment catalogue |
| WP6 — traditional software development | C++17 library, Python bridge, unit/integration tests, build instructions | Main native-code deliverable |
| WP7 — lawyer allocation/retention | no direct implementation | Requires legal case, lawyer, workload, allocation, outcome and retention data; financial prices cannot validate it |

The transferable insight is the engineering method: introduce known perturbations, preserve ground truth, capture traces, measure quality/cost, attribute degradation to mechanisms, and make evidence-based design decisions.

The analogy has limits. Incorrect prices are not semantically equivalent to false legal facts or fabricated citations. A future legal-workflow testbed would need domain-specific synthetic faults and expert-validated evaluation criteria.

---

## 16. Minimum deliverables

### 16.1 Scientific/ML deliverables for Path B

- [ ] Approved self-selected dataset(s) or documented course-staff agreement.
- [ ] Clear sample, feature, class and class-balance definitions.
- [ ] Data provenance and preprocessing description.
- [ ] At least three course methods, including an NN and tree-based ensemble.
- [ ] Method theory, cost functions, training procedure and hyperparameters.
- [ ] Implementation and verification tests.
- [ ] Systematic hyperparameter search using training/validation data.
- [ ] Same held-out test data for model comparisons.
- [ ] Accuracy and standard error, confusion matrix, per-class precision/recall/F1, ROC/AUC; additional robustness metrics where relevant.
- [ ] Literature/benchmark comparison, with comparability limits.
- [ ] Critical assessment, computational cost and limitations.

### 16.2 Course submission format

The supplied Project 3 instructions call for approximately 5,000 words, around 10–15 figures, a PDF report and a repository containing a `Code/` folder, reproducible notebooks/scripts, README, selected results and package-installation instructions. Confirm exact current Canvas instructions before submission.

### 16.3 Engineering artefacts

- [ ] C++17 fault injector and tests.
- [ ] Python interface or documented standalone fallback.
- [ ] Immutable raw data snapshots or reproducible acquisition manifest.
- [ ] Data cards and fault taxonomy.
- [ ] Experiment protocol, split manifest and trace schema.
- [ ] Machine-readable run manifests and event logs.
- [ ] Metrics/prediction tables and reproducible figures.
- [ ] README and environment file.
- [ ] `STATE_OF_WORK.md` and `DECISIONS.md`.

---

## 17. Decision gates and stop rules

Resolve these in order. Do not spend time tuning models before the earlier gates pass.

### Gate 1 — Source and data gate

Pass if:

- each source can be loaded reproducibly;
- price fields are known and not silently mixed;
- benchmark availability and date overlap are verified;
- missingness/coverage is characterised;
- immutable snapshots are identified.

If Oslo or OSEBX is not adequate, keep the US experiment primary and document the Oslo limitation rather than forcing a weak replication.

### Gate 2 — Target gate

Pass if:

- prediction time and label horizon are unambiguous;
- the training-only threshold is implemented correctly;
- no future features exist;
- positive counts and distinct event blocks are sufficient for a meaningful validation/test assessment.

If not, predefine an alternative before looking at test results or narrow the scope.

### Gate 3 — Clean pipeline gate

Pass if returns, rolling volatility, features, labels and split boundaries pass hand-worked and automated tests.

### Gate 4 — Fault injector gate

Pass if no-op, deterministic, locality, event-log and magnitude tests pass. Fault events must be traceable to actual modified values.

### Gate 5 — Model gate

Pass if all three classifiers train and evaluate on the same clean split, with saved configs and no test tuning.

### Gate 6 — Robustness gate

Pass if at least two fault families have interpretable, repeatable results and coverage losses are reported. Attempt mitigation after this gate, not before.

### Gate 7 — Submission gate

Pass if every reported figure/table is generated from retained results, all claims are supported, the repository can be reproduced, citations are verified and the LLM declaration is accurate.

---

## 18. Known issues and inherited code notes

These are observations from earlier working sessions. Verify against the current repository before editing.

1. **Kaggle loader price fallback:** current code accepts `close` as a fallback but names the field `adj_close`. Resolve explicitly.
2. **SPY missing from Kaggle panel:** expected; acquire separately and verify date alignment.
3. **OSEBX schema:** not yet verified; do not assume `Adj Close` exists.
4. **Volatility dataset function:** an earlier version contained `adj_close = adj_close.sort_index` without parentheses. It must be `adj_close = adj_close.sort_index()`.
5. **Sort before transformations:** sort the full price DataFrame before calculating returns, not only the selected benchmark Series.
6. **Duplicate volatility columns:** an earlier implementation copied `rv` into `rv_past`, creating identical columns. Keep only one canonical trailing-volatility feature unless there is a distinct alignment reason.
7. **`__main__` guard:** it must be exactly `if __name__ == "__main__":` (case-sensitive).
8. **Global plotting parameters:** do not reference `horizon` outside its scope; use an explicit constant/configuration or pass it as a parameter.
9. **Repeated inspection script:** an earlier user message contained the same Oslo inspection code twice. Maintain one canonical implementation.
10. **Import side effects:** code at the bottom of `data/oslo_prices.py` that loads data and prints an audit must be behind an `if __name__ == "__main__":` guard so importing the module only defines functions.
11. **Python environment:** user previously ran Python 3.15 and `pip install kagglehub` failed while building PyYAML because Microsoft Visual C++ Build Tools were unavailable. Confirm the active interpreter and package support; consider a dedicated virtual environment using a supported Python version. Always use `python -m pip` from the selected interpreter.
12. **Existing Oslo audit:** it flags all missing cells. Refine interpretation to separate legitimate pre-listing/post-listing absence from unexplained internal gaps.
13. **Old project draft:** earlier versions described only a five-stock Oslo panel and Project 2 sector ETF/PCA regimes. Those are no longer the controlling design. The current design is US Kaggle + SPY, and Oslo + OSEBX independently; target is a forward volatility transition; PCA/spectral features are deferred.

---

## 19. Literature and claims discipline

The project is positioned at the intersection of:

- financial data quality and outlier treatment;
- missing financial observations;
- realised-volatility forecasting and regime classification;
- structured corruption/fault robustness in time series;
- data validation, provenance and reproducible ML evaluation.

Do not rely on inherited bibliography entries without checking that each source exists and that the cited title, authors, year, venue, DOI and claim are correct. Some references in early drafts were provisional. For each source, record what it actually demonstrates and how close it is to the present question.

A focused literature search should ask:

1. Has financial time-series classification been evaluated under controlled bad-tick, stale-price, missing-block or benchmark-alignment faults?
2. Which papers study the robustness of financial ML models to data corruption, rather than ordinary market shocks?
3. Which time-series fault benchmarks already exist, and how does this project's task differ (financial equity panel, volatility-transition classification, price-level faults and traceable C++ injector)?
4. What has been published on volatility-regime transition labels, class persistence and early-warning metrics?
5. What evidence exists for cross-sectional volatility dispersion as a predictor, and how does the proposed feature differ?

Potentially adjacent works must be examined directly rather than described from their title alone. Do not claim that this project is the first, novel or unprecedented without a documented search protocol and a defensible comparison.

---

## 20. Open decisions register

| Decision | Evidence required | Priority |
|---|---|---|
| Course approval for self-selected US/Oslo datasets | Staff response / agreed scope | Immediate |
| Exact adjusted-price fields in Kaggle files | CSV schema and source documentation | Immediate |
| SPY data schema and date overlap | actual `yfinance` output and join audit | Immediate |
| Current Oslo instrument count/date coverage | current workbook audit | Immediate |
| OSEBX availability and price semantics | actual download schema and date overlap | Immediate |
| Whether Oslo target has enough events | train/validation/test event counts and blocks | Immediate |
| Final threshold percentile | training/validation feasibility, chosen before final test | High |
| Exact train/validation/test dates and purge | date index and target-window audit | High |
| Missing-data policy | source gap diagnosis and pilot feature behaviour | High |
| Fault severity grid | pilot results and realistic interpretation | High |
| Mitigation strategy | clean-trained degradation results | After robustness experiment |
| Whether cross-sectional dispersion is viable | valid cross-section size and ablation | Optional |
| Whether pybind11 is feasible | compiler/Eigen/pybind11 build test | High for C++ integration |
| Block length / bootstrap method | temporal dependence and sensitivity analysis | High before final uncertainty claims |

Every resolved item must be entered into `DECISIONS.md` with date, evidence, rationale and impact on previous experiments.

---

## 21. Final project identity and boundaries

**Working title:** *Robust Financial Time-Series Classification Under Structured Data Corruption: A Cross-Market Python/C++ Fault-Injection and Evaluation Framework.*

The intended final product is:

1. a scientifically defensible Path B classification study;
2. two independently audited financial equity panels with their own market benchmarks, subject to data feasibility;
3. a clean-reference volatility-transition target;
4. three comparably evaluated course classifiers;
5. controlled structured corruption of input prices;
6. one measured mitigation experiment;
7. a deterministic, tested C++17 fault-injection component connected to Python;
8. mechanism-level traces and reproducible experiment records;
9. a critical discussion of limitations and transfer to the internship's testing/observability work.

The project does **not** aim to build a new neural architecture, a general-purpose financial trading system, a legal research agent, a complete agentic development platform or a lawyer allocation model.

**The governing principle is:** change one mechanism at a time, preserve the clean reference, record what changed, test what should happen, measure what did happen, and explain the difference with evidence.

---

## 22. Current first action for a fresh agent

Do not start by writing a classifier or by implementing every fault type.

**First task: repository and data audit.**

1. Inspect the repository structure and identify the current canonical versions of `Kaggle_Data_download.py`, `data/oslo_prices.py`, the transition-data code and environment files.
2. Run the existing import/data-loader tests or create a small non-destructive audit script.
3. Load the Kaggle and Oslo panels and report their actual shape, date range, instruments, raw price fields and per-instrument coverage.
4. Check whether SPY and OSEBX can be acquired and aligned; report the exact schema and number of common dates.
5. Identify the adjusted/unadjusted-price ambiguity in the Kaggle loader and propose the smallest safe correction.
6. Produce a concise evidence-based `STATE_OF_WORK.md` and list any blockers.

**Do not implement target labels until the source/price semantics and coverage are understood. Do not train models until target feasibility has been established.**

