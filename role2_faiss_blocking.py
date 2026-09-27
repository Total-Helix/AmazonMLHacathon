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
            print("[!] PyTorch not found. Falling back to FAISS CPU search...")

        if not use_gpu:
            print("[!] Proceeding with optimized FAISS CPU Search (IVF)...")

        s2_s3_files = glob.glob("data/clean_data/test_source2_clean_embeddings_chunk*.npy") + \
                      glob.glob("data/clean_data/test_source3_clean_embeddings_chunk*.npy")
        s1_files = sorted(glob.glob("data/clean_data/test_source1_clean_embeddings_chunk*.npy"))

        all_I = []
        all_D = []
        
        if use_gpu:
            print(f"[*] Starting Ultra-Low Memory GPU Search for {len(s1_df)} Source 1 entities...")
            for s1_idx, s1_chunk_file in enumerate(s1_files):
                print(f"    -> Processing Query Chunk {s1_idx + 1}/{len(s1_files)}...")
                
                s1_embs = np.load(s1_chunk_file).astype('float32')
                q_tensor = torch.tensor(s1_embs, device=device)
                q_tensor = torch.nn.functional.normalize(q_tensor, p=2, dim=1)
                del s1_embs
                
                best_scores = torch.full((q_tensor.shape[0], self.top_k), -1.0, device=device)
                best_indices = torch.zeros((q_tensor.shape[0], self.top_k), dtype=torch.int64, device=device)
                global_candidate_offset = 0
                
                for s2s3_chunk_file in s2_s3_files:
                    db_embs = np.load(s2s3_chunk_file).astype('float32')
                    db_tensor = torch.tensor(db_embs, device=device)
                    db_tensor = torch.nn.functional.normalize(db_tensor, p=2, dim=1)
                    
                    similarity = torch.matmul(q_tensor, db_tensor.T)
                    
                    k = min(self.top_k, similarity.shape[1])
                    local_top_scores, local_top_indices = torch.topk(similarity, k, dim=1)
                    local_top_indices += global_candidate_offset
                    global_candidate_offset += db_tensor.shape[0]
                    
                    combined_scores = torch.cat([best_scores, local_top_scores], dim=1)
                    combined_indices = torch.cat([best_indices, local_top_indices], dim=1)
                    
                    best_scores, final_top_idx = torch.topk(combined_scores, self.top_k, dim=1)
                    best_indices = torch.gather(combined_indices, 1, final_top_idx)
                    
                    del db_embs, db_tensor, similarity, local_top_scores, local_top_indices, combined_scores, combined_indices, final_top_idx
                    torch.cuda.empty_cache()
                
                all_D.append(best_scores.cpu().numpy())
                all_I.append(best_indices.cpu().numpy())
                del q_tensor, best_scores, best_indices
                torch.cuda.empty_cache()
                gc.collect()

        else:
            # --- NUMPY CPU CHUNKED SEARCH (Fixes 16GB RAM Thrashing) ---
            print("[!] Proceeding with Ultra-Low Memory Numpy CPU Search...")
            print("[*] This entirely bypasses the FAISS 16GB memory limit by streaming chunks!")
            
            for s1_idx, s1_chunk_file in enumerate(s1_files):
                print(f"    -> Processing Query Chunk {s1_idx + 1}/{len(s1_files)}...")
                
                s1_embs = np.load(s1_chunk_file).astype('float32')
                
                # L2 Normalize Query
                norms = np.linalg.norm(s1_embs, axis=1, keepdims=True)
                norms[norms == 0] = 1e-10
                q_mat = s1_embs / norms
                del s1_embs
                
                best_scores = np.full((q_mat.shape[0], self.top_k), -1.0, dtype='float32')
                best_indices = np.zeros((q_mat.shape[0], self.top_k), dtype='int64')
                global_candidate_offset = 0
                
                for s2s3_chunk_file in s2_s3_files:
                    db_embs = np.load(s2s3_chunk_file).astype('float32')
                    
                    # L2 Normalize Candidates
                    db_norms = np.linalg.norm(db_embs, axis=1, keepdims=True)
                    db_norms[db_norms == 0] = 1e-10
                    db_mat = db_embs / db_norms
                    del db_embs
                    
                    # Cosine Similarity via Dot Product (OpenBLAS AVX2)
                    similarity = np.dot(q_mat, db_mat.T)
                    
                    # Get Top K (use argsort on negated array to get descending)
                    k = min(self.top_k, similarity.shape[1])
                    local_top_indices = np.argsort(-similarity, axis=1)[:, :k]
                    local_top_scores = np.take_along_axis(similarity, local_top_indices, axis=1)
                    
                    local_top_indices += global_candidate_offset
                    global_candidate_offset += db_mat.shape[0]
                    
                    # Merge with running best
                    combined_scores = np.concatenate([best_scores, local_top_scores], axis=1)
                    combined_indices = np.concatenate([best_indices, local_top_indices], axis=1)
                    
                    final_top_idx = np.argsort(-combined_scores, axis=1)[:, :self.top_k]
                    best_scores = np.take_along_axis(combined_scores, final_top_idx, axis=1)
                    best_indices = np.take_along_axis(combined_indices, final_top_idx, axis=1)
                    
                    del db_mat, similarity, local_top_indices, local_top_scores, combined_scores, combined_indices, final_top_idx
                
                all_D.append(best_scores)
                all_I.append(best_indices)
                del q_mat, best_scores, best_indices
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
