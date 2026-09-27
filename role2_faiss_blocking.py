import pandas as pd
import numpy as np
import faiss
import os

class FaissCandidateGenerator:
    def __init__(self, top_k=15):
        print("=== Role 2: FAISS Algorithmic Blocking Engine ===")
        self.top_k = top_k  # How many candidates to generate per Source 1 entity
        self.dimension = 384  # Size of MiniLM-L12-v2 embeddings

    def build_and_search(self):
        import glob
        import gc
        print("[*] Loading cleaned TSV data...")
        try:
            s1_df = pd.read_csv("data/clean_data/test_source1_clean.tsv", sep='\t', usecols=['entity_id'])
            s2_df = pd.read_csv("data/clean_data/test_source2_clean.tsv", sep='\t', usecols=['entity_id'])
            s3_df = pd.read_csv("data/clean_data/test_source3_clean.tsv", sep='\t', usecols=['entity_id'])
        except FileNotFoundError:
            print("❌ Error: Cleaned data missing! Did Role 1 run the GPU script yet?")
            return

        candidate_pool_df = pd.concat([s2_df, s3_df], ignore_index=True)
        del s2_df, s3_df
        gc.collect()
        
        # Check if PyTorch GPU is available
        use_gpu = False
        try:
            import torch
            if torch.cuda.is_available():
                use_gpu = True
                device = torch.device('cuda')
                print(f"[*] Detected GPU: {torch.cuda.get_device_name(0)}")
                print("[*] Activating PyTorch Chunked GPU Search (Bypassing 16GB RAM Limit)...")
        except ImportError:
            print("[!] PyTorch not found. Please run the updated MASTER_MENU.bat to install it!")
            return

        if not use_gpu:
            print("❌ Error: No GPU detected for Role 2. The 17.6GB dataset will crash your RAM.")
            print("Please ensure you run MASTER_MENU.bat to install PyTorch with CUDA!")
            return

        s2_s3_files = glob.glob("data/clean_data/test_source2_clean_embeddings_chunk*.npy") + \
                      glob.glob("data/clean_data/test_source3_clean_embeddings_chunk*.npy")
        s1_files = sorted(glob.glob("data/clean_data/test_source1_clean_embeddings_chunk*.npy"))

        all_I = []
        all_D = []
        
        print(f"[*] Starting Ultra-Low Memory GPU Search for {len(s1_df)} Source 1 entities...")
        
        # Loop through Source 1 (Queries) one chunk at a time
        for s1_idx, s1_chunk_file in enumerate(s1_files):
            print(f"    -> Processing Query Chunk {s1_idx + 1}/{len(s1_files)}...")
            
            s1_embs = np.load(s1_chunk_file).astype('float32')
            q_tensor = torch.tensor(s1_embs, device=device)
            q_tensor = torch.nn.functional.normalize(q_tensor, p=2, dim=1)
            del s1_embs
            
            # Keep a running tracker of the Best Top-K scores and indices for THIS query chunk
            best_scores = torch.full((q_tensor.shape[0], self.top_k), -1.0, device=device)
            best_indices = torch.zeros((q_tensor.shape[0], self.top_k), dtype=torch.int64, device=device)
            
            global_candidate_offset = 0
            
            # Loop through Source 2 & 3 (Candidates) one chunk at a time
            for s2s3_chunk_file in s2_s3_files:
                db_embs = np.load(s2s3_chunk_file).astype('float32')
                db_tensor = torch.tensor(db_embs, device=device)
                db_tensor = torch.nn.functional.normalize(db_tensor, p=2, dim=1)
                
                # Multiply Query Batch x Database Batch
                similarity = torch.matmul(q_tensor, db_tensor.T)
                
                # Get the Top K from THIS specific comparison
                k = min(self.top_k, similarity.shape[1])
                local_top_scores, local_top_indices = torch.topk(similarity, k, dim=1)
                
                # Offset indices so they map to the global candidate pool dataframe
                local_top_indices += global_candidate_offset
                global_candidate_offset += db_tensor.shape[0]
                
                # Concatenate the running best with this chunk's best, and keep the overall Top K
                combined_scores = torch.cat([best_scores, local_top_scores], dim=1)
                combined_indices = torch.cat([best_indices, local_top_indices], dim=1)
                
                best_scores, final_top_idx = torch.topk(combined_scores, self.top_k, dim=1)
                best_indices = torch.gather(combined_indices, 1, final_top_idx)
                
                del db_embs, db_tensor, similarity, local_top_scores, local_top_indices, combined_scores, combined_indices, final_top_idx
                torch.cuda.empty_cache()
            
            # Finished scanning all candidates for this S1 query chunk
            all_D.append(best_scores.cpu().numpy())
            all_I.append(best_indices.cpu().numpy())
            del q_tensor, best_scores, best_indices
            torch.cuda.empty_cache()
            gc.collect()

        D = np.vstack(all_D)
        I = np.vstack(all_I)

        # 4. Generate candidate_pairs.tsv format
        print("[*] Formatting results for Role 3...")
        candidate_pairs = []
        
        for i in range(len(s1_df)):
            s1_id = s1_df.iloc[i]['entity_id']
            
            # Get the actual entity_ids of the candidates using the FAISS indices
            # We filter out matches that have a terrible cosine similarity (e.g. < 0.6) to keep precision high
            good_candidates = []
            for rank, candidate_idx in enumerate(I[i]):
                similarity_score = D[i][rank]
                if similarity_score > 0.60: # Threshold to drop absolute garbage matches early
                    good_candidates.append(candidate_pool_df.iloc[candidate_idx]['entity_id'])
            
            # Join with commas as requested by Hackathon rules
            cand_string = ",".join(good_candidates)
            candidate_pairs.append({'source1_entity_id': s1_id, 'candidate_entity_ids': cand_string})

        # Save the final file
        os.makedirs("output", exist_ok=True)
        out_df = pd.DataFrame(candidate_pairs)
        out_df.to_csv("output/candidate_pairs.tsv", sep='\t', index=False)
        print(f"✅ Saved massive candidate generation to: output/candidate_pairs.tsv")
        print("🎉 Role 2 Complete! Handing off to Role 3 (Tanuj) for LightGBM Scoring.")

if __name__ == "__main__":
    blocker = FaissCandidateGenerator(top_k=15)
    blocker.build_and_search()
