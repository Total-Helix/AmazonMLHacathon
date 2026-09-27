"""
ML Challenge 2026: Data Loading and Synthetic Bootstrap Generator
Role: The Judge (Role 3: ML & Feature Engineering)

Design Rationale & Architectural Notes:
----------------------------------------
1. Strict Pairwise Isolation:
   - Feature engineering requires comparing Source 1 records against Candidate records (Source 2 and Source 3).
   - In accordance with competition rules, we construct a pairwise flattened representation:
     [source1_id, candidate_id, name_1, name_2, address_1, address_2, country_1, country_2, label]
   - Labels are binary (1 = true match, 0 = non-match distractor).

2. Synthetic Benchmark Suite:
   To guarantee our pipeline is robust before real Role 1/2 candidate outputs are ready,
   we generate a comprehensive synthetic dataset covering four mission-critical edge cases:
     a. US/India Fuzzy Variants: Legal suffix expansions ("Pvt Ltd", "Corp"), abbreviation
        transpositions, landmark insertions ("Near SBI ATM"), and phonetic spelling differences.
     b. Cross-Lingual Semantic Match (France): French source record ("Pharmacie Centrale de Paris")
        matched against English translation ("Paris Central Pharmacy"). Lexical overlap is low,
        testing dense multilingual embedding representations.
     c. Deceptive High-Lexical Non-Matches (France & US): Entities that share extensive stopword/prefix
        overlap (e.g. "Pharmacie Saint-Germain" vs. "Pharmacie Saint-Germain-des-Prés" or
        "Omega Solutions" vs. "Omega Dental Care" at the same building), requiring precise address
        and token discrimination to prevent catastrophic false merges under F_0.5 penalty.
     d. Singletons (Restraint & Empty Match Test): Source 1 records with 0 genuine matches or
        distractors that must be rejected. The output format demands that these entities map to
        an empty string `""` in `matching_results.tsv`. Correctly leaving them blank yields a
        perfect 1.0 macro score per entity.

3. Tab-Separated TSV Integrity:
   Addresses and business names frequently contain commas and quotes. All files are written
   and parsed strictly with tab delimiters (`\t`), matching the competition's exact format.
"""

import os
import csv
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import pandas as pd
import numpy as np

from src.config import CONFIG, ERConfig


# -----------------------------------------------------------------------------
# Synthetic Data Generator
# -----------------------------------------------------------------------------
class SyntheticDataGenerator:
    """
    Generates a realistic synthetic bootstrap dataset adhering to the competition schema.
    Includes the four required stress-test cases:
    1. US/India fuzzy matches
    2. French cross-lingual matches
    3. Deceptive high-overlap non-matches (hard negatives)
    4. Singletons (0 valid matches)
    """

    def __init__(self, config: ERConfig = CONFIG):
        self.config = config
        self.fake_dir = config.fake_data_dir

    def generate_all(self) -> None:
        """Generate both train and test synthetic datasets with candidate pairs."""
        train_dir = self.fake_dir / "train"
        test_dir = self.fake_dir / "test"
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        print("[SyntheticDataGenerator] Generating synthetic training set...")
        self._generate_train_split(train_dir)

        print("[SyntheticDataGenerator] Generating synthetic test set...")
        self._generate_test_split(test_dir)
        print(f"[SyntheticDataGenerator] Successfully generated synthetic datasets in: {self.fake_dir}")

    def _generate_train_split(self, target_dir: Path) -> None:
        """
        Creates synthetic training tables:
        - train_source1.tsv
        - train_source2.tsv
        - train_source3.tsv
        - train_ground_truth.tsv
        - candidate_pairs.tsv (simulating Role 2 output)
        """
        # Source 1 Reference Entities (Training covers US and India)
        s1_records = [
            # 1. US Fuzzy Variant
            {"entity_id": "S1-1001", "business_name": "ABC Tech", "business_address": "100 Innovation Way, San Jose, CA", "country": "US"},
            # 2. India Fuzzy Variant
            {"entity_id": "S1-1002", "business_name": "Shri Balaji Electronics", "business_address": "Shop 14, MG Road, Bangalore, Karnataka", "country": "India"},
            # 3. India Landmark / Transliteration Variant
            {"entity_id": "S1-1003", "business_name": "Modern Retailers Pvt Ltd", "business_address": "Plot 55, Udyog Vihar Phase 4, Gurgaon, Haryana", "country": "India"},
            # 4. US Multi-Source Matches
            {"entity_id": "S1-1004", "business_name": "Apex Logistics Inc", "business_address": "450 Industrial Parkway, Atlanta, GA", "country": "US"},
            # 5. Deceptive Non-match Target Entity (US)
            {"entity_id": "S1-1005", "business_name": "Omega Solutions LLC", "business_address": "777 Market Street, Suite 500, San Francisco, CA", "country": "US"},
            # 6. Singleton Entity 1 (India - True matches: none)
            {"entity_id": "S1-1006", "business_name": "Unique Himalayan Herbs", "business_address": "Mall Road, Manali, Himachal Pradesh", "country": "India"},
            # 7. Singleton Entity 2 (US - True matches: none)
            {"entity_id": "S1-1007", "business_name": "Desert Horizon Mining", "business_address": "8800 Copper Ridge Road, Tucson, AZ", "country": "US"},
            # 8. Standard US Match
            {"entity_id": "S1-1008", "business_name": "Green Leaf Cafe", "business_address": "12 Elm Street, Seattle, WA", "country": "US"},
            # 9. India Devanagari / Hindi Transliteration Match
            {"entity_id": "S1-1009", "business_name": "राम मार्केटिंग प्राइवेट लिमिटेड", "business_address": "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi", "country": "India"},
            # 10. US Multi-word Legal Expansion Match
            {"entity_id": "S1-1010", "business_name": "BioTech Research Labs", "business_address": "300 Science Park Drive, Cambridge, MA", "country": "US"},
            # 11. US Hard Negative (Same street, different company)
            {"entity_id": "S1-1011", "business_name": "First National Dental Clinic", "business_address": "100 Main St, Suite 10, Dallas, TX", "country": "US"},
            # 12. India Address Abbreviation Match
            {"entity_id": "S1-1012", "business_name": "Kaveri Silk Emporium", "business_address": "No 45 Commercial St, Tasker Town, Bangalore, KA", "country": "India"},
            # 13. US Abbreviation vs Full Name
            {"entity_id": "S1-1013", "business_name": "NYC Auto Works", "business_address": "550 11th Ave, New York, NY", "country": "US"},
            # 14. Singleton 3 (US)
            {"entity_id": "S1-1014", "business_name": "Lone Star Wind Energy", "business_address": "Highway 281, Abilene, TX", "country": "US"},
            # 15. India Deceptive Chain Distractor (Same brand, different branch)
            {"entity_id": "S1-1015", "business_name": "Cafe Coffee Point (Koramangala)", "business_address": "80 Feet Rd, 4th Block, Koramangala, Bengaluru, Karnataka", "country": "India"},
        ]

        # Source 2 Records
        s2_records = [
            # Match for S1-1001 (US)
            {"entity_id": "S2-2001", "business_name": "ABC Technologies Pvt Ltd", "business_address": "100 Innovation Way, Ste 200, San Jose, CA", "country": "US"},
            # Match for S1-1003 (India)
            {"entity_id": "S2-2003", "business_name": "Modern Retailers", "business_address": "Near Cyber City, Phase IV, Gurugram, HR", "country": "India"},
            # Match for S1-1004 (US)
            {"entity_id": "S2-2004", "business_name": "Apex Logistics Corporation", "business_address": "450 Industrial Pkwy, Atlanta, Georgia", "country": "US"},
            # Deceptive Distractor for S1-1005 (Different business type, same street)
            {"entity_id": "S2-2005", "business_name": "Omega Dental Care", "business_address": "777 Market Street, Suite 100, San Francisco, CA", "country": "US"},
            # Distractor candidate for Singleton S1-1006
            {"entity_id": "S2-2006", "business_name": "Himalayan Herbal Stores", "business_address": "The Mall, Shimla, Himachal Pradesh", "country": "India"},
            # Match for S1-1008
            {"entity_id": "S2-2008", "business_name": "Green Leaf Coffee & Cafe", "business_address": "12 Elm St, Seattle, Washington", "country": "US"},
            # Match for S1-1009 (English transliteration)
            {"entity_id": "S2-2009", "business_name": "Ram Marketing Private Limited", "business_address": "Khasra No 570/13, New Delhi, West Delhi", "country": "India"},
            # Match for S1-1010
            {"entity_id": "S2-2010", "business_name": "Bio-Tech Research Laboratories LLC", "business_address": "300 Science Park Dr, Cambridge, Massachusetts", "country": "US"},
            # Distractor for S1-1011 (Same address, different store)
            {"entity_id": "S2-2011", "business_name": "First National Legal Advocates", "business_address": "100 Main St, Suite 50, Dallas, TX", "country": "US"},
            # Match for S1-1012
            {"entity_id": "S2-2012", "business_name": "Kaveri Silks", "business_address": "45 Commercial Street, Tasker Town, Bengaluru, Karnataka", "country": "India"},
            # Match for S1-1013
            {"entity_id": "S2-2013", "business_name": "New York City Auto Works Inc", "business_address": "550 11th Avenue, New York, New York", "country": "US"},
            # Distractor for S1-1014 (Singleton)
            {"entity_id": "S2-2014", "business_name": "Lone Star Solar Systems", "business_address": "Highway 281, San Angelo, TX", "country": "US"},
            # Distractor for S1-1015 (Different branch / Indiranagar)
            {"entity_id": "S2-2015", "business_name": "Cafe Coffee Point (Indiranagar)", "business_address": "100 Feet Rd, Indiranagar, Bengaluru, Karnataka", "country": "India"},
        ]

        # Source 3 Records
        s3_records = [
            # Match for S1-1001 (Another match from Source 3)
            {"entity_id": "S3-3001", "business_name": "A.B.C. Tech Inc", "business_address": "100 Innovation Way, San Jose, California", "country": "US"},
            # Match for S1-1002 (India)
            {"entity_id": "S3-3002", "business_name": "Shree Balaji Electronics Pvt. Ltd.", "business_address": "Near Metro Station, M.G. Rd, Bengaluru, KA", "country": "India"},
            # Second Match for S1-1004
            {"entity_id": "S3-3004", "business_name": "Apex Logistics", "business_address": "450 Industrial Parkway, Atlanta, GA", "country": "US"},
            # Deceptive Distractor for S1-1001 (Same street name, different company)
            {"entity_id": "S3-3099", "business_name": "ABC Consulting Group", "business_address": "100 Innovation Way, San Jose, CA", "country": "US"},
            # Distractor for S1-1002 (Different store)
            {"entity_id": "S3-3098", "business_name": "Balaji Telecom", "business_address": "MG Road, Bangalore, Karnataka", "country": "India"},
            # Distractor candidate for Singleton S1-1007
            {"entity_id": "S3-3007", "business_name": "Desert Star Mining Co", "business_address": "Highway 10, Phoenix, AZ", "country": "US"},
            # Second match for S1-1010
            {"entity_id": "S3-3010", "business_name": "BioTech Laboratories", "business_address": "300 Science Park, Cambridge, MA", "country": "US"},
            # Match for S1-1015 (Koramangala branch)
            {"entity_id": "S3-3015", "business_name": "Cafe Coffee Point", "business_address": "80 Ft Road, Koramangala 4th Block, Bangalore", "country": "India"},
        ]

        # Ground Truth Matches (mapping S1 -> list of genuine S2/S3 matches)
        ground_truth = {
            "S1-1001": ["S2-2001", "S3-3001"],
            "S1-1002": ["S3-3002"],
            "S1-1003": ["S2-2003"],
            "S1-1004": ["S2-2004", "S3-3004"],
            "S1-1005": [],  # Omega Solutions has only a deceptive non-match distractor
            "S1-1006": [],  # Singleton 1 (Empty)
            "S1-1007": [],  # Singleton 2 (Empty)
            "S1-1008": ["S2-2008"],
            "S1-1009": ["S2-2009"],
            "S1-1010": ["S2-2010", "S3-3010"],
            "S1-1011": [],  # Hard negative (Empty)
            "S1-1012": ["S2-2012"],
            "S1-1013": ["S2-2013"],
            "S1-1014": [],  # Singleton 3 (Empty)
            "S1-1015": ["S3-3015"],
        }

        # Simulated Blocking / Candidate Generation output from Stage 2
        candidate_pairs = {
            "S1-1001": ["S2-2001", "S3-3001", "S3-3099"],
            "S1-1002": ["S3-3002", "S3-3098"],
            "S1-1003": ["S2-2003"],
            "S1-1004": ["S2-2004", "S3-3004"],
            "S1-1005": ["S2-2005"],
            "S1-1006": ["S2-2006"],
            "S1-1007": ["S3-3007"],
            "S1-1008": ["S2-2008"],
            "S1-1009": ["S2-2009"],
            "S1-1010": ["S2-2010", "S3-3010"],
            "S1-1011": ["S2-2011"],
            "S1-1012": ["S2-2012"],
            "S1-1013": ["S2-2013"],
            "S1-1014": ["S2-2014"],
            "S1-1015": ["S2-2015", "S3-3015"],
        }

        # Write TSVs
        self._write_source_tsv(target_dir / "train_source1.tsv", s1_records)
        self._write_source_tsv(target_dir / "train_source2.tsv", s2_records)
        self._write_source_tsv(target_dir / "train_source3.tsv", s3_records)
        self._write_gt_tsv(target_dir / "train_ground_truth.tsv", ground_truth)
        self._write_candidates_tsv(target_dir / "candidate_pairs.tsv", candidate_pairs)

    def _generate_test_split(self, target_dir: Path) -> None:
        """
        Creates synthetic test tables (US, India, and France):
        - test_source1.tsv
        - test_source2.tsv
        - test_source3.tsv
        - candidate_pairs.tsv (simulating Role 2 output on test set)
        """
        s1_test_records = [
            # 1. US Fuzzy Variant
            {"entity_id": "S1-9001", "business_name": "Falcon Dynamics Corp", "business_address": "500 Aerospace Blvd, Austin, TX", "country": "US"},
            # 2. Cross-Lingual Match (France - French source to English candidate)
            {"entity_id": "S1-9002", "business_name": "Pharmacie Centrale de Paris", "business_address": "12 Rue de Rivoli, 75001 Paris, Île-de-France", "country": "France"},
            # 3. Deceptive High-Lexical Non-Match (France - Shared prefix and stop words, distinct entity)
            {"entity_id": "S1-9003", "business_name": "Pharmacie Saint-Germain", "business_address": "45 Boulevard Saint-Germain, 75005 Paris, Île-de-France", "country": "France"},
            # 4. India Multi-source Match
            {"entity_id": "S1-9004", "business_name": "National Cyber Infotech", "business_address": "Plot 12, HITEC City, Hyderabad, Telangana", "country": "India"},
            # 5. Singleton Entity (No candidates surviving / 0 true matches)
            {"entity_id": "S1-9005", "business_name": "Bordeaux Antique Galerie", "business_address": "8 Quai des Chartrons, Bordeaux, Nouvelle-Aquitaine", "country": "France"},
            # 6. US Singleton Entity
            {"entity_id": "S1-9006", "business_name": "Solitary Pine Cabins", "business_address": "Remote Trail 4, Flagstaff, AZ", "country": "US"},
        ]

        s2_test_records = [
            # Match for S1-9001
            {"entity_id": "S2-8001", "business_name": "Falcon Dynamics Corporation", "business_address": "500 Aerospace Blvd, Ste 100, Austin, Texas", "country": "US"},
            # Cross-Lingual Match for S1-9002 (English translation)
            {"entity_id": "S2-8002", "business_name": "Paris Central Pharmacy", "business_address": "12 Rue de Rivoli, Paris", "country": "France"},
            # Deceptive non-match candidate for S1-9003
            {"entity_id": "S2-8003", "business_name": "Pharmacie Saint-Germain-des-Prés", "business_address": "10 Rue Bonaparte, 75006 Paris, Île-de-France", "country": "France"},
            # Match for S1-9004
            {"entity_id": "S2-8004", "business_name": "National Cyber Infotech Pvt Ltd", "business_address": "HITEC City Phase 2, Hyderabad, Telangana", "country": "India"},
            # Distractor for S1-9006
            {"entity_id": "S2-8006", "business_name": "Pine Ridge Cabins", "business_address": "Pine Mountain Road, Prescott, AZ", "country": "US"},
        ]

        s3_test_records = [
            # Match for S1-9001
            {"entity_id": "S3-7001", "business_name": "Falcon Dynamics", "business_address": "500 Aerospace Boulevard, Austin, TX", "country": "US"},
            # Distractor for S1-9002 (Different pharmacy in Paris)
            {"entity_id": "S3-7002", "business_name": "Pharmacie Centrale de Lyon", "business_address": "12 Rue Centrale, Lyon", "country": "France"},
            # Distractor for S1-9004
            {"entity_id": "S3-7004", "business_name": "National Infotech Services", "business_address": "Madhapur, Hyderabad, TS", "country": "India"},
        ]

        # Candidate pairs generated by blocking stage
        test_candidate_pairs = {
            "S1-9001": ["S2-8001", "S3-7001"],
            "S1-9002": ["S2-8002", "S3-7002"],  # S2 is true cross-lingual match, S3 is distractor
            "S1-9003": ["S2-8003"],              # S2 is deceptive non-match
            "S1-9004": ["S2-8004", "S3-7004"],  # S2 is true match, S3 is distractor
            "S1-9005": [],                      # Singleton with 0 candidates
            "S1-9006": ["S2-8006"],              # Singleton with weak candidate
        }

        # Write TSVs
        self._write_source_tsv(target_dir / "test_source1.tsv", s1_test_records)
        self._write_source_tsv(target_dir / "test_source2.tsv", s2_test_records)
        self._write_source_tsv(target_dir / "test_source3.tsv", s3_test_records)
        self._write_candidates_tsv(target_dir / "candidate_pairs.tsv", test_candidate_pairs)

    def _write_source_tsv(self, path: Path, records: List[Dict[str, str]]) -> None:
        """Write source entity table strictly with tab separators."""
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["entity_id", "business_name", "business_address", "country"])
            for r in records:
                writer.writerow([r["entity_id"], r["business_name"], r["business_address"], r["country"]])

    def _write_gt_tsv(self, path: Path, gt_dict: Dict[str, List[str]]) -> None:
        """Write ground truth table strictly with tab separators."""
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["source1_entity_id", "matched_entity_ids"])
            for s1, matches in gt_dict.items():
                writer.writerow([s1, ",".join(matches)])

    def _write_candidates_tsv(self, path: Path, cand_dict: Dict[str, List[str]]) -> None:
        """Write candidate pairs table strictly with tab separators."""
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["source1_entity_id", "candidate_entity_ids"])
            for s1, cands in cand_dict.items():
                writer.writerow([s1, ",".join(cands)])


# -----------------------------------------------------------------------------
# Data Loader and Pairwise Formatter
# -----------------------------------------------------------------------------
class DataLoader:
    """
    Handles loading, parsing, and pairwise expansion of TSV files
    for both training and inference pipelines.
    """

    def __init__(self, config: ERConfig = CONFIG):
        self.config = config

    @staticmethod
    def load_source_tsv(path: Path) -> pd.DataFrame:
        """
        Load a source TSV file (entity_id, business_name, business_address, country).
        Guarantees correct string dtype and handles NaN values cleanly.
        """
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {path}")
        
        df = pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            na_values=[""],
            encoding="utf-8"
        )
        # Fill any missing text with empty strings
        for col in ["business_name", "business_address", "country"]:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str).str.strip()
        return df

    @staticmethod
    def load_id_list_tsv(path: Path, value_col_name: str) -> Dict[str, List[str]]:
        """
        Load a TSV file with format: source1_entity_id \t comma_separated_ids.
        Returns a dictionary: {s1_id: [id1, id2, ...]}.
        """
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        result: Dict[str, List[str]] = {}
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            if not header or len(header) < 2:
                raise ValueError(f"Malformed TSV header in {path}: {header}")
            
            for row in reader:
                if not row:
                    continue
                s1_id = row[0].strip()
                ids_str = row[1].strip() if len(row) > 1 else ""
                ids_list = [x.strip() for x in ids_str.split(",") if x.strip()] if ids_str else []
                result[s1_id] = ids_list
        return result

    def load_split(self, split: str = "train") -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Optional[Dict[str, Set[str]]], Dict[str, List[str]]]:
        """
        Load all source tables, ground truth (if train), and candidate pairs for a given split.
        
        Returns:
            s1_df: DataFrame of Source 1 records
            target_lookup: Combined dictionary mapping entity_id -> record Dict (from Source 2 & 3)
            ground_truth: Dict mapping s1_id -> set of true matched IDs (or None for test)
            candidate_pairs: Dict mapping s1_id -> list of candidate IDs
        """
        is_train = (split.lower() == "train")
        s1_path = self.config.train_source1_path if is_train else self.config.test_source1_path
        s2_path = self.config.train_source2_path if is_train else self.config.test_source2_path
        s3_path = self.config.train_source3_path if is_train else self.config.test_source3_path
        cand_path = self.config.train_candidate_pairs_path if is_train else self.config.test_candidate_pairs_path

        s1_df = self.load_source_tsv(s1_path)
        s2_df = self.load_source_tsv(s2_path)
        s3_df = self.load_source_tsv(s3_path)

        # Build fast indexed lookups for Source 2 and Source 3
        combined_target_df = pd.concat([s2_df, s3_df], ignore_index=True)
        target_lookup = combined_target_df.set_index("entity_id").to_dict(orient="index")

        # Load ground truth if training split
        ground_truth: Optional[Dict[str, Set[str]]] = None
        if is_train:
            gt_dict = self.load_id_list_tsv(self.config.train_ground_truth_path, value_col_name="matched_entity_ids")
            ground_truth = {s1: set(matches) for s1, matches in gt_dict.items()}

        # Load or Generate candidate pairs
        if is_train:
            print("[*] Auto-generating synthetic training candidates from Ground Truth to save time...")
            candidate_pairs = {}
            target_ids = list(target_lookup.keys())
            import random
            random.seed(self.config.random_seed) # Deterministic
            for s1, matches in ground_truth.items():
                # 1 positive match + 4 random negative matches per query
                cands = list(matches)
                cands.extend(random.sample(target_ids, min(4, len(target_ids))))
                candidate_pairs[s1] = cands
        else:
            candidate_pairs = self.load_id_list_tsv(cand_path, value_col_name="candidate_entity_ids")

        return s1_df, target_lookup, ground_truth, candidate_pairs

    def build_pairwise_dataframe(
        self,
        s1_df: pd.DataFrame,
        target_lookup: Dict[str, Dict[str, str]],
        candidate_pairs: Dict[str, List[str]],
        ground_truth: Optional[Dict[str, Set[str]]] = None
    ) -> pd.DataFrame:
        """
        Expands candidate pair lists into a flattened pairwise DataFrame for ML modeling.
        
        Columns produced:
        - source1_id: ID of Source 1 entity
        - cand_id: ID of Candidate entity (from S2 or S3)
        - name_1: Cleaned business name of S1
        - name_2: Cleaned business name of Candidate
        - addr_1: Cleaned address of S1
        - addr_2: Cleaned address of Candidate
        - country_1: Country of S1
        - country_2: Country of Candidate
        - label: Binary target (1 = match, 0 = non-match distractor; NaN if ground_truth is None)
        """
        # --- FAST VECTORIZED APPROACH: Build pairs via pandas merge instead of Python loops ---
        # Build a flat (s1_id, cand_id) table from the dict first
        s1_ids_flat = []
        cand_ids_flat = []
        for s1_id, cand_ids in candidate_pairs.items():
            s1_ids_flat.extend([s1_id] * len(cand_ids))
            cand_ids_flat.extend(cand_ids)

        pairs_flat = pd.DataFrame({"source1_id": s1_ids_flat, "cand_id": cand_ids_flat})

        # Build S1 lookup dataframe
        s1_info_df = s1_df[["entity_id", "business_name", "business_address", "country"]].rename(columns={
            "entity_id": "source1_id", "business_name": "name_1",
            "business_address": "addr_1", "country": "country_1"
        })

        # Build target lookup dataframe
        if target_lookup:
            tgt_rows = [{"cand_id": cid, "business_name": v.get("business_name", ""),
                         "business_address": v.get("business_address", ""), "country": v.get("country", "")}
                        for cid, v in target_lookup.items()]
            tgt_df = pd.DataFrame(tgt_rows).rename(columns={
                "business_name": "name_2", "business_address": "addr_2", "country": "country_2"
            })
        else:
            tgt_df = pd.DataFrame(columns=["cand_id", "name_2", "addr_2", "country_2"])

        # Merge S1 info, then target info — vectorized, no Python loop over rows
        pairwise_df = pairs_flat.merge(s1_info_df, on="source1_id", how="left")
        pairwise_df = pairwise_df.merge(tgt_df, on="cand_id", how="left")

        # Fill NaN strings
        for col in ["name_1", "name_2", "addr_1", "addr_2", "country_1", "country_2"]:
            pairwise_df[col] = pairwise_df[col].fillna("").astype(str)

        # Assign labels if ground truth is available
        if ground_truth is not None:
            gt_set = {s1: list(matches) for s1, matches in ground_truth.items()}
            gt_flat = pd.DataFrame([
                {"source1_id": s1, "cand_id": cid}
                for s1, cids in gt_set.items() for cid in cids
            ])
            if not gt_flat.empty:
                gt_flat["label"] = 1
                pairwise_df = pairwise_df.merge(gt_flat, on=["source1_id", "cand_id"], how="left")
                pairwise_df["label"] = pairwise_df["label"].fillna(0).astype(int)
            else:
                pairwise_df["label"] = 0
        else:
            pairwise_df["label"] = None

        if pairwise_df.empty:
            pairwise_df = pd.DataFrame(columns=[
                "source1_id", "cand_id", "name_1", "name_2",
                "addr_1", "addr_2", "country_1", "country_2", "label"
            ])
        return pairwise_df

