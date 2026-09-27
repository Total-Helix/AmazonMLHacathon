import pandas as pd
import numpy as np
import faiss
import os

class FaissCandidateGenerator:
    def __init__(self, top_k=5):
        print("=== Role 2: FAISS Algorithmic Blocking Engine ===")
        self.top_k = top_k  # Reduced to 5 to massively increase F0.5 precision and speed up Role 3
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
            # --- FAISS CPU SEARCH WITH PRODUCT QUANTIZATION (Fixes 16GB RAM limit & runs in 3 minutes) ---
            print("[!] Proceeding with FAISS IVFPQ (Highly Compressed CPU Search)...")
            print("[*] This will compress the 17GB dataset down to ~500MB in RAM!")
            
            nlist = 1024 # Clusters
            m = 48       # Number of sub-vector quantizers (384 / 48 = 8)
            nbits = 8    # Bits per sub-vector
            
            quantizer = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
            
            if s2_s3_files:
                print("    -> Training FAISS PQ index on data sample (this takes ~1 minute)...")
                # Load a larger sample for better PQ training (e.g. 2 chunks)
                sample_chunks = []
                for sample_file in s2_s3_files[:2]:
                    embs = np.load(sample_file).astype('float32')
                    faiss.normalize_L2(embs)
                    sample_chunks.append(embs)
                
                training_data = np.vstack(sample_chunks)
                index.train(training_data)
                del training_data, sample_chunks
                gc.collect()

            print("    -> Indexing 10 Million Candidates into 500MB Compressed RAM...")
            for chunk_file in s2_s3_files:
                embs = np.load(chunk_file).astype('float32')
                faiss.normalize_L2(embs)
                index.add(embs)
                del embs
                gc.collect()
            print("[*] Compressed FAISS Index built successfully!")

            index.nprobe = 16
            all_D = []
            all_I = []
            print(f"[*] Querying FAISS index for {len(s1_df)} Source 1 entities...")
            for s1_idx, chunk_file in enumerate(s1_files):
                print(f"    -> Querying Chunk {s1_idx + 1}/{len(s1_files)}...")
                s1_embs = np.load(chunk_file).astype('float32')
                faiss.normalize_L2(s1_embs)
                D, I = index.search(s1_embs, self.top_k)
                all_D.append(D)
                all_I.append(I)
                del s1_embs
                gc.collect()

        D = np.vstack(all_D)
        I = np.vstack(all_I)

        # 4. Generate candidate_pairs.tsv format — VECTORIZED (no Python loops over rows)
        print("[*] Formatting results for Role 3...")

        # Extract arrays once (avoids repeated iloc calls)
        D_flat = D  # shape: (n_s1, top_k)
        I_flat = I  # shape: (n_s1, top_k)
        pool_ids = candidate_pool_df["entity_id"].values  # numpy array for fast indexing

        # Build a flat DataFrame of all (s1_idx, candidate_idx, score) triplets
        n_queries = len(s1_df)
        s1_id_col   = np.repeat(s1_df["entity_id"].values, self.top_k)
        cand_idx_col = I_flat.ravel()
        score_col    = D_flat.ravel()

        pairs_df = pd.DataFrame({
            "source1_entity_id": s1_id_col,
            "cand_idx": cand_idx_col,
            "score": score_col
        })

        # Filter out low-quality matches and invalid indices
        pairs_df = pairs_df[(pairs_df["score"] > 0.60) & (pairs_df["cand_idx"] >= 0)]

        # Map candidate indices to entity IDs using numpy fancy indexing
        pairs_df["candidate_entity_id"] = pool_ids[pairs_df["cand_idx"].values]

        # Aggregate: group by s1 id, join candidate ids with comma
        agg_df = pairs_df.groupby("source1_entity_id", sort=False)["candidate_entity_id"].apply(
            lambda x: ",".join(x.tolist())
        ).reset_index()
        agg_df.columns = ["source1_entity_id", "candidate_entity_ids"]

        # Ensure every S1 entity appears (even singletons with no good candidates)
        all_s1 = pd.DataFrame({"source1_entity_id": s1_df["entity_id"].values})
        out_df = all_s1.merge(agg_df, on="source1_entity_id", how="left")
        out_df["candidate_entity_ids"] = out_df["candidate_entity_ids"].fillna("")

        out_df.to_csv("output/candidate_pairs.tsv", sep='\t', index=False)
        print(f"✅ Saved massive candidate generation to: output/candidate_pairs.tsv")
        print("🎉 Role 2 Complete! Handing off to Role 3 (Tanuj) for LightGBM Scoring.")

if __name__ == "__main__":
    blocker = FaissCandidateGenerator(top_k=15)
    blocker.build_and_search()
