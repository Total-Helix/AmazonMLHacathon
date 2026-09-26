# Amazon ML Challenge 2026: Business Entity Resolution Pipeline
**Role:** The Judge (Role 3: ML & Feature Engineering)  
**Target Metric:** Macro $F_{0.5}$ (Precision weighted 2× over Recall)

---

## 1. Executive Summary & Design Overview

This codebase implements a high-throughput, leakage-free Entity Resolution matching engine ("The Judge"). Operating as Stage 3 in the competition pipeline, it receives shortlisted candidate pairs from Stages 1 & 2 (Cleaning and Candidate Generation/Blocking), constructs multi-tiered pairwise feature representations, trains a calibrated gradient boosted decision tree classifier (LightGBM), and optimizes decision thresholds directly against the competition's macro-averaged $F_{0.5}$ metric.

---

## 2. Directory Structure

```
ML_2026_Challenge/
├── data/
│   └── fake_data/                  # Synthetic bootstrap testbed
│       ├── train/                  # Synthetic train tables (S1, S2, S3, GT, candidates)
│       └── test/                   # Synthetic test tables (S1, S2, S3, candidates)
├── models/
│   ├── lightgbm_er_judge.joblib    # Serialized LightGBM classifier
│   ├── tfidf_vectorizers.joblib    # Character n-gram TF-IDF vectorizers
│   └── optimal_threshold.json      # Optimal F_0.5 decision threshold & metadata
├── output/
│   ├── matching_results.tsv        # Scored leaderboard submission (TAB-separated)
│   └── candidate_pairs.tsv         # Final candidate set fed to model (TAB-separated)
├── src/
│   ├── __init__.py
│   ├── config.py                   # Centralized configuration & 1-line dataset toggle
│   ├── data_loading.py             # Data loading & synthetic benchmark generator
│   ├── features.py                 # Multi-tier pairwise feature engineering engine
│   ├── train.py                    # Leakage-free GroupK training & F_0.5 threshold tuner
│   └── inference.py                # Batch scoring, constraint enforcement & validator
├── student_resource/               # Real competition data, templates & validation tools
│   ├── dataset/
│   │   ├── train/                  # Real training source tables & ground truth
│   │   └── test/                   # Real test source tables
│   └── utils/
│       └── validate_submission.py  # Official competition submission validator
├── main.py                         # End-to-end execution entry point
├── requirements.txt                # Pinned dependencies
└── README.md
```

---

## 3. Real Competition Data Inspection

Extracted directly from `student_resource.zip`:

### Source Tables (`*_source1.tsv`, `*_source2.tsv`, `*_source3.tsv`)
- **Columns:** `entity_id`, `business_name`, `business_address`, `country`
- **Format:** Tab-separated (`\t`), UTF-8 encoded.
- **Observations:**
  - Training entities span `US` and `India`.
  - Test entities span `US`, `India`, and `France` (introducing cross-lingual matching and accented characters).
  - Heavy noise patterns: Punctuation prefixes (`<< Team Ecole`, `-- Holloway Peak`), Devanagari Hindi text (`राम मार्केटिंग प्राइवेट लिमिटेड`), landmark address descriptions (`Near Metro Station, M.G. Rd`).

### Ground Truth Table (`train_ground_truth.tsv`)
- **Columns:** `source1_entity_id`, `matched_entity_ids`
- **Format:** `source1_entity_id\tS2-XXXXX,S3-YYYYY` (or `source1_entity_id\t` for singletons with zero matches).

---

## 4. Feature Engineering Architecture (`src/features.py`)

All features are calculated strictly **pairwise** between Source 1 and Candidate entities without dataset leakage:

1. **Fine-Grained Lexical Metrics (RapidFuzz C++ core):**
   - `name_jaro_winkler`: Robust to legal suffixes ("ABC Tech" vs "ABC Technologies Pvt Ltd").
   - `name_token_sort_ratio`: Invariant to token transposition ("Tech ABC" vs "ABC Tech").
   - `name_token_set_ratio`: Handles subset containment and missing tokens.
   - `name_levenshtein_ratio` & `name_partial_ratio`: Substring alignment and normalized edit distance.
   - Parallel suite of lexical similarity metrics for `business_address`.

2. **Subword Character N-Gram TF-IDF Cosine Similarities (2–4 chars):**
   - Robust against typos, inflectional suffixes, French diacritics (`é` vs `e`), and Hindi transliteration discrepancies.
   - Pairwise cosine similarities computed for Name, Address, and Full Concatenated Record (`Name + " " + Address`).

3. **Multilingual Semantic Embeddings:**
   - Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (117M params, Apache 2.0).
   - Maps French entities (e.g., `Pharmacie Centrale de Paris` <-> `Paris Central Pharmacy`) and multilingual aliases into a shared 384-dimensional vector space.
   - Deduplicated batch encoding for latency and memory optimization.

4. **Structural & Domain Discrepancy Indicators:**
   - Address digit / PIN code / house number Jaccard overlap ratio (`addr_digits_overlap`).
   - Length ratios and absolute length differences for name and address.
   - Exact string match flags and primary brand first-token match flag.
   - Country concordance indicator (`country_match`).

---

## 5. Metric Alignment & Threshold Tuning (`src/train.py`)

- **Metric:** Macro $F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$.
- **Entity-Grouped Split:** Partitioning is grouped strictly by `source1_entity_id` so that all candidates for a given entity remain together, preventing leakage.
- **Singleton Handling:** A Source 1 entity with zero true matches earns a score of 1.0 when predicted empty, and 0.0 if any false match is predicted.
- **Precision Bias:** Because precision is weighted 2× over recall ($\beta=0.5$), the optimal decision threshold $\theta^*$ is determined via systematic grid sweep in $[0.50, 0.99]$.

---

## 6. How to Run the Pipeline

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run on Synthetic Bootstrap Dataset
```bash
python main.py --use-fake-data --train --infer
```

### Step 3: Run on Real Competition Dataset
Toggle in `src/config.py` by setting `USE_FAKE_DATA = False` or pass the CLI argument:
```bash
python main.py --use-real-data --train --infer
```

### Step 4: Validate Generated Submission Files
The pipeline automatically runs the official validator, or you can invoke it manually:
```bash
python student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir data/fake_data/test
```
