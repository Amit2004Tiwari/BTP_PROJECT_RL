"""
Deep Reinforcement Learning Classifier for Book Review Content Detection
Precompute Embeddings Script - Optimized with GPU Support and Path Consistency

Author: Research Project
Python Version: 3.12.11
System: Linux (SSH Compatible)
"""

import os
import torch
import pickle
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer

# Import configuration for path consistency
import sys
sys.path.append(str(Path(__file__).parent))
from config import config

# -----------------------------
# CONFIG - Using paths from config.py
# -----------------------------
DATA_FILE = config.DATASET_PATH
EMBED_DIR = config.PROJECT_ROOT / "data" / "embeddings"
EMBED_MODEL = config.PRIMARY_ENCODER  # Use model from config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Create embeddings directory
os.makedirs(EMBED_DIR, exist_ok=True)

print(f"🚀 Using device: {DEVICE}")
print(f"📂 Data file: {DATA_FILE}")
print(f"📂 Embeddings directory: {EMBED_DIR}")
print(f"🤖 Model: {EMBED_MODEL}")

# -----------------------------
# LOAD DATA WITH ERROR HANDLING
# -----------------------------
print("📊 Loading dataset...")

try:
    df = pd.read_csv(DATA_FILE)
    print(f"✅ Loaded dataset: {len(df)} rows")
except FileNotFoundError:
    raise FileNotFoundError(f"❌ Dataset not found at {DATA_FILE}")

# Validate required columns
required_columns = ['description_text', 'review_text']
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"❌ Missing required columns: {missing_cols}")

# Handle NaN values
print("🔧 Preprocessing data...")
df['description_text'] = df['description_text'].fillna('').astype(str)
df['review_text'] = df['review_text'].fillna('').astype(str)

descriptions = df['description_text'].tolist()
reviews = df['review_text'].tolist()

print(f"📝 Prepared {len(descriptions)} descriptions and {len(reviews)} reviews")

# -----------------------------
# INITIALIZE MODEL WITH GPU SUPPORT
# -----------------------------
print(f"🤖 Loading model: {EMBED_MODEL}")
model = SentenceTransformer(EMBED_MODEL, device=DEVICE)

# Verify embedding dimension matches config
actual_dim = model.get_sentence_embedding_dimension()
expected_dim = config.EMBEDDING_DIMENSION

if actual_dim != expected_dim:
    print(f"⚠️  Warning: Model dimension {actual_dim} != config dimension {expected_dim}")
    print(f"   Updating config expectation to {actual_dim}")

print(f"✅ Model loaded successfully. Embedding dimension: {actual_dim}")

# -----------------------------
# HELPER TO COMPUTE EMBEDDINGS WITH GPU SUPPORT
# -----------------------------
def compute_embeddings(text_list, batch_size=64):
    """Compute embeddings with GPU support and proper memory management"""
    embeddings = []
    
    print(f"🔄 Computing embeddings in batches of {batch_size}...")
    
    for i in tqdm(range(0, len(text_list), batch_size), desc="Processing batches"):
        batch = text_list[i:i+batch_size]
        
        # Encode on GPU/CPU, then move to CPU for storage
        with torch.no_grad():  # Save memory
            emb = model.encode(
                batch, 
                convert_to_tensor=True, 
                device=DEVICE,
                show_progress_bar=False  # Use tqdm instead
            )
            embeddings.append(emb.cpu())  # Move to CPU for concatenation
    
    # Concatenate all embeddings
    final_embeddings = torch.cat(embeddings, dim=0)
    return final_embeddings

# -----------------------------
# COMPUTE & SAVE EMBEDDINGS
# -----------------------------
print("\n📝 Encoding description embeddings...")
description_embeddings = compute_embeddings(descriptions)
torch.save(description_embeddings, EMBED_DIR / "description_embeddings.pt")
print(f"💾 Saved description embeddings: {description_embeddings.shape}")

print("\n📝 Encoding review embeddings...")
review_embeddings = compute_embeddings(reviews)
torch.save(review_embeddings, EMBED_DIR / "review_embeddings.pt")
print(f"💾 Saved review embeddings: {review_embeddings.shape}")

# -----------------------------
# SAVE METADATA & INDEX MAPPING
# -----------------------------
print("\n💾 Saving metadata and index mapping...")

index_mapping = {i: idx for i, idx in enumerate(df.index.tolist())}
metadata = {
    "num_samples": len(df),
    "embedding_dim": description_embeddings.shape[1],
    "model_name": EMBED_MODEL,
    "device_used": DEVICE,
    "description_shape": list(description_embeddings.shape),
    "review_shape": list(review_embeddings.shape)
}

# Save using pathlib paths for consistency
with open(EMBED_DIR / "index_mapping.pkl", "wb") as f:
    pickle.dump(index_mapping, f)

with open(EMBED_DIR / "metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("✅ Metadata and index mapping saved successfully!")

# -----------------------------
# VERIFICATION - CHECK ALL FILES
# -----------------------------
files = ["description_embeddings.pt", "review_embeddings.pt", "index_mapping.pkl", "metadata.pkl"]

print(f"\n📂 Verifying all saved files in {EMBED_DIR}...")
for filename in files:
    filepath = EMBED_DIR / filename
    
    if not filepath.exists():
        print(f"❌ Missing file: {filename}")
        continue
    
    if filename.endswith(".pt"):
        obj = torch.load(filepath, map_location="cpu")
        print(f"✅ {filename} | shape: {obj.shape} | dtype: {obj.dtype} | size: {obj.numel():,} elements")
    else:
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        content_preview = str(obj) if len(str(obj)) < 100 else f"{type(obj)} with {len(obj)} items"
        print(f"✅ {filename} | type: {type(obj).__name__} | content: {content_preview}")

# -----------------------------
# FINAL SUMMARY
# -----------------------------
print(f"\n🎉 All embeddings and metadata saved successfully!")
print(f"📊 Summary:")
print(f"   • Dataset rows: {len(df):,}")
print(f"   • Embedding dimension: {description_embeddings.shape[1]}")
print(f"   • Description embeddings: {description_embeddings.shape}")
print(f"   • Review embeddings: {review_embeddings.shape}")
print(f"   • Device used: {DEVICE}")
print(f"   • Model: {EMBED_MODEL}")
print(f"   • Files saved to: {EMBED_DIR}")

# Cleanup GPU memory if using CUDA
if DEVICE == "cuda":
    torch.cuda.empty_cache()
    print("🧹 GPU memory cleared")

print("✅ Precompute embeddings script completed successfully!")
