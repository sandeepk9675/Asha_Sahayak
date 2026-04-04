# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Ingest Knowledge Base
# MAGIC Parse medical guidelines → chunk → embed via API → build FAISS index on DBFS.

# COMMAND ----------

import sys, os
os.environ["ASHA_DELTA_BASE"] = "/dbfs/asha_sahayak/delta"
os.environ["ASHA_FAISS_PATH"] = "/dbfs/asha_sahayak/faiss"

notebook_dir = os.getcwd()
for candidate in [notebook_dir, os.path.dirname(notebook_dir), "/Workspace/Repos/asha-sahayak"]:
    if os.path.isdir(os.path.join(candidate, "src")):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
        break

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Load and Parse Knowledge Base Documents

# COMMAND ----------

import glob
import json

# Path to knowledge base files
KB_DIR = os.path.join(os.path.dirname(notebook_dir) if "notebooks" in notebook_dir else notebook_dir, "data", "knowledge_base")
# Fallback: try workspace path
if not os.path.isdir(KB_DIR):
    KB_DIR = "/Workspace/Repos/asha-sahayak/data/knowledge_base"

print(f"Knowledge base directory: {KB_DIR}")

documents = []

# Load text files
for filepath in glob.glob(os.path.join(KB_DIR, "*.txt")):
    filename = os.path.basename(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    documents.append({
        "source": filename,
        "content": content,
    })
    print(f"Loaded: {filename} ({len(content)} chars)")

print(f"\nTotal documents: {len(documents)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Chunk Documents

# COMMAND ----------

def chunk_text(text, chunk_size=500, overlap=100):
    """Split text into overlapping chunks."""
    chunks = []
    sentences = text.replace("\n\n", "\n").split("\n")
    
    current_chunk = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Keep overlap
            words = current_chunk.split()
            overlap_words = words[-overlap // 5:] if len(words) > overlap // 5 else words
            current_chunk = " ".join(overlap_words) + " " + sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


# Chunk all documents
all_chunks = []
for doc in documents:
    chunks = chunk_text(doc["content"], chunk_size=500, overlap=100)
    for i, chunk in enumerate(chunks):
        all_chunks.append({
            "text": chunk,
            "source": doc["source"],
            "chunk_id": i,
        })

print(f"Total chunks: {len(all_chunks)}")
for doc in documents:
    source_chunks = [c for c in all_chunks if c["source"] == doc["source"]]
    print(f"  {doc['source']}: {len(source_chunks)} chunks")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Generate Embeddings

# COMMAND ----------

from src.api.embeddings_client import get_passage_embeddings
import numpy as np

# Embed all chunks (in batches to avoid API limits)
BATCH_SIZE = 32
all_embeddings = []

texts = [c["text"] for c in all_chunks]

for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i:i + BATCH_SIZE]
    embeddings = get_passage_embeddings(batch)
    all_embeddings.append(embeddings)
    print(f"Embedded batch {i // BATCH_SIZE + 1}/{(len(texts) - 1) // BATCH_SIZE + 1}")

embeddings_matrix = np.vstack(all_embeddings)
print(f"\nEmbeddings shape: {embeddings_matrix.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Build FAISS Index

# COMMAND ----------

import faiss

# Create FAISS index (Inner Product for cosine similarity on normalized vectors)
dim = embeddings_matrix.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings_matrix)

print(f"FAISS index built: {index.ntotal} vectors, dimension {dim}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Save to DBFS

# COMMAND ----------

FAISS_DIR = os.environ.get("ASHA_FAISS_PATH", "/dbfs/asha_sahayak/faiss")
os.makedirs(FAISS_DIR, exist_ok=True)

# Save FAISS index
index_path = os.path.join(FAISS_DIR, "index.faiss")
faiss.write_index(index, index_path)
print(f"FAISS index saved to: {index_path}")

# Save chunk metadata
meta_path = os.path.join(FAISS_DIR, "chunks_metadata.json")
with open(meta_path, "w") as f:
    json.dump(all_chunks, f, indent=2)
print(f"Metadata saved to: {meta_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Verify Index

# COMMAND ----------

# Test search
from src.api.embeddings_client import get_embeddings

test_query = "What are the danger signs in pregnancy that need immediate referral?"
query_vec = get_embeddings([test_query])

distances, indices = index.search(query_vec, 3)

print(f"Query: {test_query}\n")
for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    chunk = all_chunks[idx]
    print(f"\nResult {i+1} (score: {dist:.4f}):")
    print(f"Source: {chunk['source']}")
    print(f"Text: {chunk['text'][:200]}...")

print("\n✅ Knowledge base ingestion complete!")
