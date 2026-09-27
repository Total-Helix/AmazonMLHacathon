import pandas as pd
import torch
import re
import os
import gc
from sentence_transformers import SentenceTransformer

class GPUDatasetPreprocessor:
    def __init__(self):
        print("=== Role 1: GPU Preprocessing Engine ===")
        
        # 1. Check for RTX 5050 (CUDA)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[*] Detected Device: {self.device.upper()}")
        
        if self.device == "cpu":
            print("⚠️ WARNING: No GPU detected! This will be extremely slow. Ensure PyTorch with CUDA is installed.")
        else:
            print(f"[*] GPU Name: {torch.cuda.get_device_name(0)}")
            
        # 2. Load Multilingual Model (Optimized for RTX Tensor Cores)
        print("[*] Loading Multilingual Embedding Model to VRAM (FP16 Mode)...")
        # Using FP16 (Half Precision) doubles the speed on RTX cards and halves VRAM usage
        self.model = SentenceTransformer(
            'paraphrase-multilingual-MiniLM-L12-v2', 
            device=self.device,
            model_kwargs={"torch_dtype": torch.float16}
        )

    def normalize_text(self, text):
        """Role 1 Core Task: Standardize text, legal suffixes, and French characters."""
        if pd.isna(text):
            return ""
            
        text = str(text).lower().strip()
        
        # Resolve Legal Suffixes
        legal_fixes = {
            r'\bcorp\b': 'corporation',
            r'\bpvt\b': 'private',
            r'\bltd\b': 'limited',
            r'\binc\b': 'incorporated',
            r'\bllc\b': 'limited liability company'
        }
        for pattern, repl in legal_fixes.items():
            text = re.sub(pattern, repl, text)
            
        # Handle French/Special characters gracefully
        # (The multilingual model handles accents naturally, so we just remove rogue punctuation)
        text = re.sub(r'[^\w\sàâçéèêëîïôûùüÿñæœ]', '', text) 
        
        # Remove extra whitespace
        text = " ".join(text.split())
        return text

    def process_file_in_chunks(self, file_path, output_path, chunk_size=100000):
        """Processes massive TSV files without crashing the 8GB VRAM RTX 5050."""
        import numpy as np
        print(f"\n[*] Processing: {file_path}")
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return
            
        chunk_iter = pd.read_csv(file_path, sep='\t', chunksize=chunk_size)
        
        is_first_chunk = True
        
        for i, chunk in enumerate(chunk_iter):
            print(f"    -> Cleaning Chunk {i+1} ({len(chunk)} rows)...")
            
            # 1. Clean the text (CPU)
            chunk['clean_name'] = chunk['business_name'].apply(self.normalize_text)
            chunk['clean_address'] = chunk['business_address'].apply(self.normalize_text)
            
            # 2. Generate GPU Embeddings (RTX 5050 Heavy Lift)
            # Since we are using FP16, we can safely bump batch_size to 1024 to max out CUDA cores
            print(f"    -> Generating GPU Embeddings (Batch Size 1024)...")
            combined_text = (chunk['clean_name'] + " " + chunk['clean_address']).tolist()
            
            embeddings = self.model.encode(
                combined_text, 
                batch_size=1024, 
                show_progress_bar=False, 
                device=self.device
            )
            
            # SAVE EMBEDDINGS IMMEDIATELY TO DISK TO PREVENT RAM CRASH
            chunk_npy_path = output_path.replace('.tsv', f'_embeddings_chunk{i}.npy')
            np.save(chunk_npy_path, embeddings)
            
            # Save the cleaned text TSV
            mode = 'w' if is_first_chunk else 'a'
            header = is_first_chunk
            chunk.to_csv(output_path, sep='\t', index=False, mode=mode, header=header)
            
            is_first_chunk = False
            
            # Free RAM & VRAM
            del embeddings
            del combined_text
            torch.cuda.empty_cache()
            gc.collect()
            
        print(f"✅ Saved cleaned dataset to: {output_path}")

if __name__ == "__main__":
    preprocessor = GPUDatasetPreprocessor()
    
    # Ensure output directory exists
    os.makedirs("data/clean_data", exist_ok=True)
    
    # Process the real datasets (Assuming they are extracted in student_resource)
    files_to_process = [
        ("student_resource/dataset/train/train_source1.tsv", "data/clean_data/train_source1_clean.tsv"),
        ("student_resource/dataset/train/train_source2.tsv", "data/clean_data/train_source2_clean.tsv"),
        ("student_resource/dataset/train/train_source3.tsv", "data/clean_data/train_source3_clean.tsv"),
        
        ("student_resource/dataset/test/test_source1.tsv", "data/clean_data/test_source1_clean.tsv"),
        ("student_resource/dataset/test/test_source2.tsv", "data/clean_data/test_source2_clean.tsv"),
        ("student_resource/dataset/test/test_source3.tsv", "data/clean_data/test_source3_clean.tsv")
    ]
    
    for input_file, output_file in files_to_process:
        preprocessor.process_file_in_chunks(input_file, output_file)
        
    print("\n🎉 GPU Preprocessing Complete! Handing off to Role 2 for Blocking.")
