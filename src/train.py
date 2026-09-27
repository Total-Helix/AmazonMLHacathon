"""
ML Challenge 2026: Model Training & F0.5 Threshold Optimizer
Role: The Judge (Role 3: ML & Feature Engineering)

Design Rationale & Architectural Notes:
----------------------------------------
1. Entity-Grouped Cross-Validation (Leakage Prevention):
   - In entity resolution, splitting pairwise rows at random introduces severe data leakage:
     candidates for the same Source 1 entity would leak across train and validation folds.
   - We enforce strict grouping by `source1_id`: all candidates for a given Source 1 entity
     exist exclusively within either the training partition or the validation partition.

2. Exact Competition Metric Alignment (Macro F_0.5 with Singletons):
   - The competition metric is not pairwise binary F1, nor is it micro-averaged F_0.5.
   - It is Macro-Averaged F_0.5 per Source 1 entity across all evaluated entities:
     * F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
     * For singletons (true_matches = empty):
       - If predicted_matches = empty -> score = 1.0 (True Negative credit)
       - If predicted_matches != empty -> score = 0.0 (False Positive penalty)
     * For non-singletons (true_matches != empty):
       - If predicted_matches = empty -> score = 0.0
       - Otherwise standard F_0.5 on sets.

3. Threshold Optimization Strategy:
   - Standard classification models default to threshold = 0.50, which optimizes for balanced F1.
   - Because F_0.5 penalizes false merges twice as heavily as false negatives, the optimal
     operating point will naturally shift higher (often 0.70 - 0.90+).
   - We systematically sweep thresholds in [0.50, 0.99] with step 0.01 on validation predictions
     to select the exact cutoff maximizing Macro F_0.5.

4. Model Selection (LightGBM):
   - Tree-based gradient boosting (LightGBM) is chosen for its fast training on millions of rows,
     native handling of non-linear similarity combinations, resistance to multicollinearity
     between string similarity metrics, and monotonic probability calibration.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit

from src.config import CONFIG, ERConfig
from src.data_loading import DataLoader, SyntheticDataGenerator
from src.features import FeatureEngineeringEngine


# -----------------------------------------------------------------------------
# Metric Evaluator: Macro F0.5 per Entity
# -----------------------------------------------------------------------------
def calculate_entity_f05(
    true_matches: Set[str],
    pred_matches: Set[str],
    beta: float = 0.5
) -> float:
    """
    Computes F_beta score for a single Source 1 entity following competition rules.
    Singletons:
      - true empty & pred empty => 1.0
      - true empty & pred non-empty => 0.0
    Non-singletons:
      - true non-empty & pred empty => 0.0
      - true non-empty & pred non-empty => F_beta
    """
    if len(true_matches) == 0 and len(pred_matches) == 0:
        return 1.0
    if len(true_matches) == 0 and len(pred_matches) > 0:
        return 0.0
    if len(true_matches) > 0 and len(pred_matches) == 0:
        return 0.0

    tp = len(true_matches & pred_matches)
    fp = len(pred_matches - true_matches)
    fn = len(true_matches - pred_matches)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision == 0.0 and recall == 0.0:
        return 0.0

    beta_sq = beta ** 2
    f_score = ((1 + beta_sq) * precision * recall) / (beta_sq * precision + recall)
    return float(f_score)


def evaluate_macro_f05(
    all_s1_ids: List[str],
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    beta: float = 0.5
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluates Macro F_0.5 across all Source 1 entities in the evaluation partition.
    Returns:
      macro_score: Mean F_0.5 score across all entities
      per_entity_scores: Dict mapping s1_id -> score
    """
    scores = {}
    for s1_id in all_s1_ids:
        true_set = ground_truth.get(s1_id, set())
        pred_set = predictions.get(s1_id, set())
        scores[s1_id] = calculate_entity_f05(true_set, pred_set, beta=beta)

    macro_score = float(np.mean(list(scores.values()))) if scores else 0.0
    return macro_score, scores


# -----------------------------------------------------------------------------
# Training & Optimization Pipeline
# -----------------------------------------------------------------------------
class Trainer:
    """Orchestrates model training, cross-validation, and F_0.5 threshold optimization."""

    def __init__(self, config: ERConfig = CONFIG):
        self.config = config
        self.feature_engine = FeatureEngineeringEngine(config=config)
        self.model: Optional[lgb.LGBMClassifier] = None
        self.optimal_threshold: float = 0.50
        self.feature_names: List[str] = []

    def run(self) -> Dict[str, Any]:
        """
        Executes end-to-end training and optimization:
        1. Loads training split data and candidate pairs.
        2. Performs Grouped Train/Validation split by source1_id.
        3. Fits feature engineering engine and extracts features.
        4. Trains LightGBM model.
        5. Sweeps decision threshold in [0.50, 0.99] to maximize Macro F_0.5 on validation split.
        6. Serializes model, vectorizers, and metadata to disk.
        """
        loader = DataLoader(config=self.config)

        # Generate fake data if running in fake mode and files do not exist
        if self.config.use_fake_data:
            if not self.config.train_source1_path.exists():
                print("[Trainer] Generating synthetic bootstrap data...")
                SyntheticDataGenerator(config=self.config).generate_all()

        print("[Trainer] Loading training data...")
        s1_df, target_lookup, ground_truth, candidate_pairs = loader.load_split("train")

        print(f"[Trainer] Building pairwise training DataFrame from {len(candidate_pairs)} S1 candidate lists...")
        pairwise_df = loader.build_pairwise_dataframe(s1_df, target_lookup, candidate_pairs, ground_truth)
        print(f"[Trainer] Total pairwise comparisons: {len(pairwise_df)} (Positives: {pairwise_df['label'].sum()}, Negatives: {len(pairwise_df) - pairwise_df['label'].sum()})")

        # Entity-grouped Train / Validation Split
        all_s1_ids = s1_df["entity_id"].unique()
        
        # --- MASSIVE SPEEDUP: Downsample to 50,000 queries for training ---
        # Training LightGBM on 16 million pairs is unnecessary and freezes the CPU during TF-IDF fitting.
        import numpy as np
        if len(all_s1_ids) > 50000:
            print(f"[Trainer] Downsampling training dataset from {len(all_s1_ids)} to 50,000 queries to prevent CPU freeze...")
            np.random.seed(self.config.random_seed)
            sampled_s1_ids = np.random.choice(all_s1_ids, size=50000, replace=False)
            pairwise_df = pairwise_df[pairwise_df["source1_id"].isin(sampled_s1_ids)].copy()
            all_s1_ids = sampled_s1_ids

        print(f"[Trainer] Splitting {len(all_s1_ids)} unique S1 entities into Train / Validation (val_size={self.config.val_size})...")

        gss = GroupShuffleSplit(n_splits=1, test_size=self.config.val_size, random_state=self.config.random_seed)
        train_idx, val_idx = next(gss.split(pairwise_df, groups=pairwise_df["source1_id"]))

        train_pairs = pairwise_df.iloc[train_idx].reset_index(drop=True)
        val_pairs = pairwise_df.iloc[val_idx].reset_index(drop=True)

        val_s1_ids = set(val_pairs["source1_id"].unique())
        # Include any validation S1 entities that had 0 candidate pairs in the candidate pool
        train_s1_ids_in_pairs = set(train_pairs["source1_id"].unique())
        for s1 in all_s1_ids:
            if s1 not in train_s1_ids_in_pairs and s1 not in val_s1_ids:
                val_s1_ids.add(s1)
        val_s1_list = list(val_s1_ids)

        print(f"[Trainer] Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)} across {len(val_s1_list)} Val entities.")

        # 1. Fit feature engineer on training pairs
        print("[Trainer] Fitting FeatureEngineeringEngine on training corpus...")
        self.feature_engine.fit(train_pairs)

        # 2. Extract features
        print("[Trainer] Extracting features for training pairs...")
        X_train = self.feature_engine.transform(train_pairs)
        y_train = train_pairs["label"].astype(int).values
        self.feature_names = list(X_train.columns)

        print("[Trainer] Extracting features for validation pairs...")
        X_val = self.feature_engine.transform(val_pairs)
        y_val = val_pairs["label"].astype(int).values

        # 3. Train LightGBM Classifier
        print("[Trainer] Training LightGBM Judge Model...")
        self.model = lgb.LGBMClassifier(**self.config.lgbm_params)
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )

        # 4. Predict probabilities on validation set
        val_probs = self.model.predict_proba(X_val)[:, 1]
        val_pairs = val_pairs.copy()
        val_pairs["pred_prob"] = val_probs

        # 5. Optimize Threshold for Macro F0.5
        print("[Trainer] Optimizing decision threshold for Macro F_0.5 on validation entities...")
        best_threshold, best_f05, threshold_log = self._optimize_threshold(
            val_pairs, val_s1_list, ground_truth
        )

        self.optimal_threshold = best_threshold
        print(f"\n=======================================================")
        print(f" [Trainer Results] Optimal Threshold: {best_threshold:.2f}")
        print(f" [Trainer Results] Best Validation Macro F_0.5: {best_f05:.4f}")
        print(f"=======================================================\n")

        # 6. Feature Importance Logging
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importances_
        }).sort_values(by="importance", ascending=False)
        print("[Trainer] Top 10 Feature Importances:")
        print(importance_df.head(10).to_string(index=False))

        # 7. Save Model, Threshold, and Feature Vectorizers
        self.save_artifacts(best_threshold, best_f05)

        return {
            "optimal_threshold": best_threshold,
            "best_macro_f05": best_f05,
            "threshold_sweep": threshold_log,
            "feature_importance": importance_df.to_dict(orient="records")
        }

    def _optimize_threshold(
        self,
        val_pairs_df: pd.DataFrame,
        val_s1_ids: List[str],
        ground_truth: Dict[str, Set[str]]
    ) -> Tuple[float, float, List[Dict[str, float]]]:
        """
        Sweeps threshold from threshold_min to threshold_max with step threshold_step.
        Evaluates full Macro F_0.5 across all validation S1 entities (including singletons).
        """
        thresholds = np.arange(
            self.config.threshold_min,
            self.config.threshold_max + 1e-5,
            self.config.threshold_step
        )

        best_score = -1.0
        best_threshold = 0.50
        sweep_log = []

        # Group candidate predictions by source1_id for fast filtering
        val_grouped = {s1: group for s1, group in val_pairs_df.groupby("source1_id")}

        for th in thresholds:
            th = round(float(th), 3)
            pred_dict: Dict[str, Set[str]] = {}

            for s1_id in val_s1_ids:
                if s1_id in val_grouped:
                    sub_df = val_grouped[s1_id]
                    surviving = sub_df[sub_df["pred_prob"] >= th]["cand_id"].tolist()
                    pred_dict[s1_id] = set(surviving)
                else:
                    pred_dict[s1_id] = set()

            macro_f05, _ = evaluate_macro_f05(val_s1_ids, ground_truth, pred_dict, beta=self.config.beta)
            sweep_log.append({"threshold": th, "macro_f05": macro_f05})

            if macro_f05 > best_score:
                best_score = macro_f05
                best_threshold = th

        return best_threshold, best_score, sweep_log

    def save_artifacts(self, optimal_threshold: float, best_macro_f05: float) -> None:
        """Serializes trained model, feature extraction engine, and metadata to disk."""
        self.config.model_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save LightGBM model
        joblib.dump(self.model, self.config.model_save_path)
        print(f"[Trainer] Saved LightGBM model to {self.config.model_save_path}")

        # 2. Save Feature Engineering Vectorizers
        self.feature_engine.save(self.config.vectorizer_save_path)

        # 3. Save Optimal Threshold & Metadata
        metadata = {
            "optimal_threshold": optimal_threshold,
            "best_macro_f05": best_macro_f05,
            "feature_names": self.feature_names,
            "beta": self.config.beta,
            "embedding_model": self.config.embedding_model_name,
            "use_fake_data": self.config.use_fake_data
        }
        with open(self.config.threshold_save_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        print(f"[Trainer] Saved threshold metadata to {self.config.threshold_save_path}")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()
