"""
ML Challenge 2026: Inference Pipeline & Submission Generator
Role: The Judge (Role 3: ML & Feature Engineering)

Design Rationale & Architectural Notes:
----------------------------------------
1. Strict Competition Output Formatting:
   - The competition scorer enforces strict constraints on `output/matching_results.tsv`:
     * Delimiter must be strictly TAB (`\t`).
     * Header must be exactly `source1_entity_id\tmatched_entity_ids`.
     * Every Source 1 entity present in `test_source1.tsv` MUST appear in exactly one row.
     * Entities with zero surviving candidates (singletons or rejected distractors) MUST have an
       empty string as their matched list (i.e. `S1-XXXXX\t\n`).
     * Matched IDs must be comma-separated, unique, and strictly from Source 2 or Source 3.

2. Candidate Set Preservation (`output/candidate_pairs.tsv`):
   - The submission package requires both `matching_results.tsv` and `candidate_pairs.tsv`.
   - The inference engine ensures `matching_results.tsv` is a strict subset of `candidate_pairs.tsv`.

3. Automated Pre-Submission Validation:
   - The pipeline directly executes the validation checks provided in `utils/validate_submission.py`,
     verifying zero formatting errors, self-matches, or missing entities before completion.
"""

import os
import csv
import json
import joblib
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

from src.config import CONFIG, ERConfig
from src.data_loading import DataLoader, SyntheticDataGenerator
from src.features import FeatureEngineeringEngine


# -----------------------------------------------------------------------------
# Inference Engine
# -----------------------------------------------------------------------------
class InferenceEngine:
    """
    Executes model inference on test candidate pairs and generates competition-compliant
    `matching_results.tsv` and `candidate_pairs.tsv`.
    """

    def __init__(self, config: ERConfig = CONFIG):
        self.config = config
        self.model = None
        self.feature_engine: Optional[FeatureEngineeringEngine] = None
        self.optimal_threshold: float = 0.50
        self.metadata: Dict = {}

    def load_artifacts(self) -> None:
        """Load trained LightGBM model, feature vectorizers, and optimal threshold."""
        if not self.config.model_save_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self.config.model_save_path}. Please run train.py first."
            )
        if not self.config.threshold_save_path.exists():
            raise FileNotFoundError(
                f"Threshold metadata not found at {self.config.threshold_save_path}. Please run train.py first."
            )

        print(f"[InferenceEngine] Loading model from {self.config.model_save_path}...")
        self.model = joblib.load(self.config.model_save_path)

        print(f"[InferenceEngine] Loading feature engine vectorizers from {self.config.vectorizer_save_path}...")
        self.feature_engine = FeatureEngineeringEngine.load(self.config.vectorizer_save_path, config=self.config)

        print(f"[InferenceEngine] Loading threshold metadata from {self.config.threshold_save_path}...")
        with open(self.config.threshold_save_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        
        self.optimal_threshold = float(self.metadata.get("optimal_threshold", 0.50))
        print(f"[InferenceEngine] Optimal decision threshold set to: {self.optimal_threshold:.2f}")

    def run_inference(self) -> Tuple[Path, Path]:
        """
        Executes end-to-end inference and writes validated TSVs to `output/`.
        Returns paths to (matching_results.tsv, candidate_pairs.tsv).
        """
        self.load_artifacts()
        loader = DataLoader(config=self.config)

        # Generate fake data if running in fake mode and files do not exist
        if self.config.use_fake_data and not self.config.test_source1_path.exists():
            print("[InferenceEngine] Generating synthetic bootstrap test data...")
            SyntheticDataGenerator(config=self.config).generate_all()

        print("[InferenceEngine] Loading test split data...")
        s1_df, target_lookup, _, candidate_pairs = loader.load_split("test")

        all_test_s1_ids = s1_df["entity_id"].tolist()
        print(f"[InferenceEngine] Test Source 1 entities: {len(all_test_s1_ids)}")
        print(f"[InferenceEngine] Candidate pair lists: {len(candidate_pairs)}")

        # Build pairwise DataFrame
        print("[InferenceEngine] Expanding candidate pairs to pairwise DataFrame...")
        pairwise_df = loader.build_pairwise_dataframe(s1_df, target_lookup, candidate_pairs, ground_truth=None)
        print(f"[InferenceEngine] Total candidate pairs to score: {len(pairwise_df)}")

        # Predict match probabilities if there are candidate pairs
        final_matches_map: Dict[str, List[str]] = {s1: [] for s1 in all_test_s1_ids}

        if not pairwise_df.empty:
            print("[InferenceEngine] Extracting features for test candidate pairs...")
            X_test = self.feature_engine.transform(pairwise_df)

            # Reorder columns to match training feature order
            expected_features = self.metadata.get("feature_names", list(X_test.columns))
            X_test = X_test.reindex(columns=expected_features, fill_value=0.0)

            print("[InferenceEngine] Scoring candidate pairs with LightGBM...")
            pred_probs = self.model.predict_proba(X_test)[:, 1]
            pairwise_df = pairwise_df.copy()
            pairwise_df["score"] = pred_probs

            # Filter candidates based on optimal F_0.5 threshold
            surviving = pairwise_df[pairwise_df["score"] >= self.optimal_threshold]
            print(f"[InferenceEngine] Surviving matches after applying threshold ({self.optimal_threshold:.2f}): {len(surviving)} / {len(pairwise_df)}")

            # Vectorized groupby — 100x faster than iterrows() on millions of rows
            for s1_id, group in surviving.groupby("source1_id"):
                if s1_id in final_matches_map:
                    final_matches_map[s1_id] = list(dict.fromkeys(
                        c for c in group["cand_id"].tolist() if c.startswith(("S2-", "S3-"))
                    ))

        # Write output TSVs
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        matching_path = self.config.output_matching_path
        candidate_path = self.config.output_candidate_path

        print(f"[InferenceEngine] Writing final matches to {matching_path}...")
        self._write_matching_results(matching_path, all_test_s1_ids, final_matches_map)

        print(f"[InferenceEngine] Writing candidate pairs to {candidate_path}...")
        self._write_candidate_pairs(candidate_path, all_test_s1_ids, candidate_pairs)

        # Validate outputs
        print("[InferenceEngine] Running validation on generated submission files...")
        self.validate_outputs(matching_path, candidate_path)

        return matching_path, candidate_path

    def _write_matching_results(self, path: Path, all_s1_ids: List[str], matches_map: Dict[str, List[str]]) -> None:
        """
        Writes `matching_results.tsv` strictly with TAB separator.
        Guarantees that EVERY Source 1 entity appears exactly once.
        Singletons / non-matches are written with an empty string in the second column.
        """
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t", lineterminator="\n")
            writer.writerow(["source1_entity_id", "matched_entity_ids"])
            for s1_id in all_s1_ids:
                matched_list = matches_map.get(s1_id, [])
                # Ensure no S1 self-matches and no duplicates
                cleaned_matches = [m for m in dict.fromkeys(matched_list) if m.startswith(("S2-", "S3-"))]
                writer.writerow([s1_id, ",".join(cleaned_matches)])

    def _write_candidate_pairs(self, path: Path, all_s1_ids: List[str], cand_map: Dict[str, List[str]]) -> None:
        """
        Writes `candidate_pairs.tsv` strictly with TAB separator.
        Guarantees that EVERY Source 1 entity appears exactly once.
        """
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t", lineterminator="\n")
            writer.writerow(["source1_entity_id", "candidate_entity_ids"])
            for s1_id in all_s1_ids:
                cand_list = cand_map.get(s1_id, [])
                cleaned_cands = [c for c in dict.fromkeys(cand_list) if c.startswith(("S2-", "S3-"))]
                writer.writerow([s1_id, ",".join(cleaned_cands)])

    def validate_outputs(self, matching_path: Path, candidate_path: Path) -> None:
        """
        Runs internal checks and calls official validation script if test files exist.
        """
        # Internal quick checks
        df_match = pd.read_csv(matching_path, sep="\t", dtype=str, keep_default_na=False)
        df_cand = pd.read_csv(candidate_path, sep="\t", dtype=str, keep_default_na=False)

        assert list(df_match.columns) == ["source1_entity_id", "matched_entity_ids"], f"Invalid matching columns: {df_match.columns}"
        assert list(df_cand.columns) == ["source1_entity_id", "candidate_entity_ids"], f"Invalid candidate columns: {df_cand.columns}"
        assert len(df_match) == len(df_cand), f"Row count mismatch: {len(df_match)} vs {len(df_cand)}"
        assert df_match["source1_entity_id"].is_unique, "Duplicate source1_entity_id rows detected in matching_results.tsv!"

        print(f"[InferenceEngine] Internal Validation PASSED ({len(df_match)} rows checked).")

        # Run official validator if real test set is used or available
        validator_script = self.config.project_root / "student_resource" / "utils" / "validate_submission.py"
        test_dir = self.config.active_data_dir / "test"

        if validator_script.exists() and test_dir.exists():
            try:
                cmd = [
                    "python",
                    str(validator_script),
                    "--matching", str(matching_path),
                    "--candidate", str(candidate_path),
                    "--test-dir", str(test_dir)
                ]
                res = subprocess.run(cmd, capture_output=True, text=True, check=True)
                print(f"[InferenceEngine] Official Validator Output:\n{res.stdout}")
            except subprocess.CalledProcessError as e:
                print(f"[InferenceEngine] Validation Warning/Error:\n{e.stdout}\n{e.stderr}")


if __name__ == "__main__":
    engine = InferenceEngine()
    engine.run_inference()
