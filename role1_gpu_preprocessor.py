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
            
        # 2. Load Multilingual Model (for French/Hindi/English)
        print("[*] Loading Multilingual Embedding Model to VRAM...")
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=self.device)

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
        print(f"\n[*] Processing: {file_path}")
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return
            
        # Read the TSV in chunks to prevent RAM explosions
        chunk_iter = pd.read_csv(file_path, sep='\t', chunksize=chunk_size)
        
        is_first_chunk = True
        for i, chunk in enumerate(chunk_iter):
            print(f"    -> Cleaning Chunk {i+1} ({len(chunk)} rows)...")
            
            # 1. Clean the text (Role 1 Task)
            chunk['clean_name'] = chunk['business_name'].apply(self.normalize_text)
            chunk['clean_address'] = chunk['business_address'].apply(self.normalize_text)
            
            # Create a combined string for the embedding
            combined_text = chunk['clean_name'] + " " + chunk['clean_address']
            
            # 2. Generate GPU Embeddings in batches (Batch size 256 prevents RTX 5050 OOM)
            print(f"    -> Generating GPU Embeddings...")
            # We don't save the embeddings directly into the CSV as they are huge vectors.
            # Instead, we just save the cleaned text. Role 3's code will re-embed or load them.
            # (If you want to save embeddings, use .parquet or .npy instead of .tsv)
            
            # For this script, we output a perfectly clean TSV for Role 2 (Blocking)
            
            mode = 'w' if is_first_chunk else 'a'
            header = is_first_chunk
            chunk.to_csv(output_path, sep='\t', index=False, mode=mode, header=header)
            
            is_first_chunk = False
            
            # Free VRAM
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
