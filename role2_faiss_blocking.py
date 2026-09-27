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
            s1_df = pd.read_csv("data/clean_data/test_source1_clean.tsv", sep='\t')
            s2_df = pd.read_csv("data/clean_data/test_source2_clean.tsv", sep='\t')
            s3_df = pd.read_csv("data/clean_data/test_source3_clean.tsv", sep='\t')
        except FileNotFoundError:
            print("❌ Error: Cleaned data missing! Did Role 1 run the GPU script yet?")
            return

        candidate_pool_df = pd.concat([s2_df, s3_df], ignore_index=True)
        
        # Check if PyTorch GPU is available
        use_gpu = False
        try:
            import torch
            if torch.cuda.is_available():
                use_gpu = True
                device = torch.device('cuda')
                print(f"[*] Detected GPU: {torch.cuda.get_device_name(0)}")
                print("[*] Activating PyTorch GPU Acceleration (Bypassing FAISS CPU)...")
        except ImportError:
            pass

        s2_s3_files = glob.glob("data/clean_data/test_source2_clean_embeddings_chunk*.npy") + \
                      glob.glob("data/clean_data/test_source3_clean_embeddings_chunk*.npy")
        s1_files = sorted(glob.glob("data/clean_data/test_source1_clean_embeddings_chunk*.npy"))

        all_D, all_I = [], []

        if use_gpu:
            # --- PYTORCH GPU SEARCH ---
            print("[*] Loading candidate pool directly into VRAM...")
            db_chunks = []
            for chunk_file in s2_s3_files:
                embs = np.load(chunk_file).astype('float32')
                # Move to GPU and normalize
                t = torch.tensor(embs, device=device)
                t = torch.nn.functional.normalize(t, p=2, dim=1)
                db_chunks.append(t)
                del embs
            
            db_tensor = torch.cat(db_chunks, dim=0).T # Transpose for matrix multiplication
            del db_chunks
            gc.collect()
            print("[*] Candidate pool loaded to GPU!")

            print(f"[*] Querying GPU for {len(s1_df)} Source 1 entities...")
            for chunk_file in s1_files:
                s1_embs = np.load(chunk_file).astype('float32')
                q_tensor = torch.tensor(s1_embs, device=device)
                q_tensor = torch.nn.functional.normalize(q_tensor, p=2, dim=1)
                
                # Cosine similarity via Matrix Multiplication
                similarity = torch.matmul(q_tensor, db_tensor)
                
                # Get Top K
                top_scores, top_indices = torch.topk(similarity, self.top_k, dim=1)
                all_D.append(top_scores.cpu().numpy())
                all_I.append(top_indices.cpu().numpy())
                
                del q_tensor, similarity, top_scores, top_indices, s1_embs
                torch.cuda.empty_cache()
                gc.collect()
                
            del db_tensor
            torch.cuda.empty_cache()

        else:
            # --- FAISS CPU SEARCH ---
            print("[*] Building FAISS Index incrementally to save RAM...")
            index = faiss.IndexFlatIP(self.dimension)
            for chunk_file in s2_s3_files:
                embs = np.load(chunk_file).astype('float32')
                faiss.normalize_L2(embs)
                index.add(embs)
                del embs
                gc.collect()
            print("[*] FAISS Index built successfully!")

            print(f"[*] Querying FAISS index for {len(s1_df)} Source 1 entities...")
            for chunk_file in s1_files:
                s1_embs = np.load(chunk_file).astype('float32')
                faiss.normalize_L2(s1_embs)
                D, I = index.search(s1_embs, self.top_k)
                all_D.append(D)
                all_I.append(I)
                del s1_embs
                gc.collect()

        # Stack the results safely
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
