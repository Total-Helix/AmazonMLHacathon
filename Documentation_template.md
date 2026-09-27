# Amazon ML Challenge 2026 - Entity Resolution Methodology

## 1. Methodology Used
Our overarching methodology treats the Business Entity Resolution problem as a massive-scale Information Retrieval (IR) and Pointwise Learning-to-Rank classification pipeline. Because the dataset entails resolving millions of entities (1.7M queries against 10M+ candidates) where standard pairwise comparisons scale quadratically ($O(N^2)$), we decomposed the pipeline into a strict two-stage system:
1. **Algorithmic Blocking / Candidate Generation:** A high-recall retrieval stage using highly optimized GPU matrix chunking and Product Quantization to reduce the search space from 17 Trillion comparisons down to just 8.6 Million highly probable candidate pairs (Top 5 per query).
2. **Machine Learning Verification (The Judge):** A high-precision LightGBM classification tree utilizing advanced Lexical (RapidFuzz), Subword (TF-IDF), and Semantic (Sentence-Transformers) feature engineering to confidently filter false positives and maximize the $F_{0.5}$ score.

## 2. Candidate Generation / Blocking Strategy
Handling the 17.6 GB of raw embedding data strictly required bypassing the 16 GB hardware RAM ceiling of our development environments. We implemented two robust strategies in `role2_faiss_blocking.py`:

**Strategy A: Ultra-Low Memory PyTorch Matrix Chunking (GPU)**
Instead of relying on FAISS (which forcefully allocates the entire vector index in RAM), we engineered a double-loop streaming algorithm. The system streams tiny 200MB chunks of embeddings from the NVMe SSD directly into the RTX 5050 GPU VRAM. It computes the dense vector inner-products using PyTorch (`torch.matmul`) with a batch size of 512, retains the Top-K indices using `torch.topk`, and flushes the VRAM. This maintains 100% Exact Search recall while utilizing less than 2 GB of system RAM.

**Strategy B: FAISS Product Quantization (`IndexIVFPQ`) (CPU Fallback)**
To guarantee execution on standard CPU hardware without thrashing, we leveraged FAISS with Inverted File Indexing and Product Quantization (IVFPQ). We compressed the 384-dimension `float32` embeddings into 48-byte codes (nbits=8), shrinking the 17 GB dataset down to ~500 MB. Combined with 1024 Voronoi clusters, this strategy algebraically skips 98% of the distance calculations.

To aggressively target the $F_{0.5}$ metric (which weights Precision 2x over Recall), we intentionally constrained our generator to `top_k=5`. This physically blocks the "long tail" of false positives (ranks 6-15) from ever reaching the LightGBM classifier.

## 3. Model Architecture and Feature Engineering
Our downstream architecture relies on a highly-tuned LightGBM Gradient Boosting Decision Tree (GBDT) designed specifically for tabular pairwise similarity scoring. 

**Feature Engineering:**
Rather than feeding raw text into the model, we extract 15+ continuous distance metrics:
* **Neural Semantics:** We utilize `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` to compute cosine similarities. This model natively maps 50+ languages to a unified dense space, bridging the gap between variations like "Pharmacie Centrale" (FR) and "Central Pharmacy" (EN).
* **Lexical Distances:** We employ `rapidfuzz` (implemented via C++ for extreme speed) to calculate Jaro-Winkler, Token Sort Ratio, and Levenshtein distances on both Name and Address fields.
* **Character N-Grams (TF-IDF):** We trained a character-level TF-IDF vectorizer. By breaking names into subword combinations (e.g., `pha`, `har`, `arm`), we successfully catch localized typos and phonetic misspellings that dense semantic embeddings occasionally blur together.
* **Domain Rules:** Exact match flags, country match flags, and address-digit-overlap Jaccard indices (e.g. comparing PIN codes / house numbers).

**Model Tuning (F0.5 Optimization):**
LightGBM was configured with `n_estimators=500`, `num_leaves=63`, and a slowed `learning_rate=0.03` to force deep, careful interaction learning. Crucially, rather than accepting the default `0.5` binary threshold, our pipeline loops over a threshold sweep (`0.50` to `0.99` in increments of `0.01`) during cross-validation. It dynamically selects the exact mathematical cutoff that achieves the highest possible macro $F_{0.5}$ score on the validation holdout, guaranteeing maximum leaderboard performance.

## 4. Execution / Hardware Constraints
The pipeline is orchestrated by a unified `MASTER_MENU.bat` which automatically detects `python3.11`/`python3.12` installations to bypass compatibility bugs with Python 3.14 on Windows. It dynamically installs the necessary NVIDIA CUDA drivers (cu121) and seamlessly handles package dependencies.
