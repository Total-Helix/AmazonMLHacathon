import pandas as pd
from rapidfuzz import fuzz, distance
import numpy as np

class FeatureEngineer:
    def __init__(self, s1_path, s2_path, s3_path):
        """Loads all source datasets into memory for fast lookup."""
        print("Loading source datasets...")
        self.s1_df = pd.read_csv(s1_path, sep='\t').set_index('entity_id')
        self.s2_df = pd.read_csv(s2_path, sep='\t').set_index('entity_id')
        self.s3_df = pd.read_csv(s3_path, sep='\t').set_index('entity_id')

    def get_record(self, entity_id):
        """Helper to fetch a record regardless of its source."""
        if str(entity_id).startswith("S1-"):
            return self.s1_df.loc[entity_id]
        elif str(entity_id).startswith("S2-"):
            return self.s2_df.loc[entity_id]
        elif str(entity_id).startswith("S3-"):
            return self.s3_df.loc[entity_id]
        return None

    def safe_str(self, val):
        return str(val).lower() if pd.notna(val) else ""

    def extract_features(self, candidate_pairs_path):
        """
        Takes the candidate_pairs.tsv and generates ML features for every candidate pair.
        Returns a DataFrame ready for the ML model (Role 3).
        """
        print(f"Reading candidates from {candidate_pairs_path}...")
        candidates_df = pd.read_csv(candidate_pairs_path, sep='\t')
        
        feature_rows = []
        
        for _, row in candidates_df.iterrows():
            s1_id = row['source1_entity_id']
            cand_ids_str = row['candidate_entity_ids']
            
            if pd.isna(cand_ids_str) or not cand_ids_str.strip():
                continue # No candidates for this entity
                
            s1_record = self.get_record(s1_id)
            s1_name = self.safe_str(s1_record['business_name'])
            s1_addr = self.safe_str(s1_record['business_address'])
            
            # Split the comma-separated candidates
            cand_ids = [c.strip() for c in cand_ids_str.split(',')]
            
            for cand_id in cand_ids:
                cand_record = self.get_record(cand_id)
                cand_name = self.safe_str(cand_record['business_name'])
                cand_addr = self.safe_str(cand_record['business_address'])
                
                # 1. Advanced String Distances - Names
                name_lev = fuzz.ratio(s1_name, cand_name) / 100.0
                name_jaro = distance.JaroWinkler.normalized_similarity(s1_name, cand_name)
                name_token_set = fuzz.token_set_ratio(s1_name, cand_name) / 100.0
                
                # 2. Advanced String Distances - Addresses
                addr_lev = fuzz.ratio(s1_addr, cand_addr) / 100.0
                addr_jaro = distance.JaroWinkler.normalized_similarity(s1_addr, cand_addr)
                addr_token_set = fuzz.token_set_ratio(s1_addr, cand_addr) / 100.0
                
                # 3. Exact matches
                exact_country = int(self.safe_str(s1_record['country']) == self.safe_str(cand_record['country']))
                
                feature_rows.append({
                    'source1_entity_id': s1_id,
                    'candidate_entity_id': cand_id,
                    'name_levenshtein': name_lev,
                    'name_jaro_winkler': name_jaro,
                    'name_token_set': name_token_set,
                    'addr_levenshtein': addr_lev,
                    'addr_jaro_winkler': addr_jaro,
                    'addr_token_set': addr_token_set,
                    'exact_country_match': exact_country
                })
                
        features_df = pd.DataFrame(feature_rows)
        print(f"Generated features for {len(features_df)} pairs.")
        return features_df

if __name__ == "__main__":
    # Test with the mock data we generated
    fe = FeatureEngineer(
        'dataset/test/test_source1.tsv',
        'dataset/test/test_source2.tsv',
        'dataset/test/test_source3.tsv'
    )
    
    # Generate the feature matrix for Role 3
    ml_features = fe.extract_features('output/candidate_pairs.tsv')
    print(ml_features.head())
    
    # Save it so Role 3 can load it directly into their XGBoost/PyTorch model
    ml_features.to_csv('output/ml_features.csv', index=False)
    print("Saved to output/ml_features.csv")
