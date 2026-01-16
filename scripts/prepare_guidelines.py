#!/usr/bin/env python3
"""
Prepare Clinical Guidelines Embeddings

One-time script to pre-compute embeddings for guideline chunks.
Run this locally or on any machine with sentence-transformers installed.

Output:
- data/guidelines/embeddings.npz - Pre-computed embeddings

Usage:
    python scripts/prepare_guidelines.py

Requirements:
    pip install sentence-transformers numpy

This script should be run once to generate embeddings. The embeddings
can then be uploaded to Kaggle as a dataset for use in notebooks.
"""

import json
import numpy as np
from pathlib import Path


def main():
    # Setup paths
    base_path = Path(__file__).parent.parent / "data" / "guidelines"
    chunks_path = base_path / "chunks.json"
    embeddings_path = base_path / "embeddings.npz"

    print("=" * 60)
    print("Clinical Guidelines Embedding Preparation")
    print("=" * 60)

    # Load chunks
    print(f"\nLoading chunks from {chunks_path}...")
    with open(chunks_path, 'r') as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} guideline chunks")

    # Load sentence-transformers
    print("\nLoading sentence-transformers model...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded: all-MiniLM-L6-v2")
    except ImportError:
        print("ERROR: sentence-transformers not installed")
        print("Install with: pip install sentence-transformers")
        return

    # Prepare texts for embedding
    # Combine relevant fields for better retrieval
    texts = []
    for chunk in chunks:
        # Create a rich text representation for embedding
        text = f"{chunk['condition']} {chunk['guideline_name']} {chunk['section']}: {chunk['content']}"
        texts.append(text)

    print(f"\nPrepared {len(texts)} texts for embedding")

    # Generate embeddings
    print("\nGenerating embeddings...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # For cosine similarity
    )

    print(f"Generated embeddings with shape: {embeddings.shape}")

    # Save embeddings
    print(f"\nSaving embeddings to {embeddings_path}...")
    np.savez_compressed(
        embeddings_path,
        embeddings=embeddings,
    )

    # Verify save
    file_size = embeddings_path.stat().st_size / 1024
    print(f"Saved! File size: {file_size:.1f} KB")

    # Quick test
    print("\n" + "=" * 60)
    print("Quick Retrieval Test")
    print("=" * 60)

    test_queries = [
        "chest pain evaluation acute coronary syndrome",
        "pneumonia treatment antibiotics",
        "heart failure diagnosis BNP",
        "COPD exacerbation management",
        "lung nodule follow-up",
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        query_embedding = model.encode(query, convert_to_numpy=True)
        query_norm = query_embedding / np.linalg.norm(query_embedding)

        similarities = np.dot(embeddings, query_norm)
        top_idx = np.argmax(similarities)

        print(f"  Top match: {chunks[top_idx]['guideline_name']} - {chunks[top_idx]['section']}")
        print(f"  Similarity: {similarities[top_idx]:.3f}")
        print(f"  Condition: {chunks[top_idx]['condition']}")

    print("\n" + "=" * 60)
    print("Embedding preparation complete!")
    print("=" * 60)
    print(f"\nFiles created:")
    print(f"  - {embeddings_path}")
    print(f"\nNext steps:")
    print("  1. Upload data/guidelines/ to Kaggle as a dataset")
    print("  2. Add dataset to your notebook inputs")
    print("  3. Use GuidelinesAgent in your pipeline")


if __name__ == "__main__":
    main()
