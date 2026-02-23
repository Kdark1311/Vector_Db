from embedding_service import EmbeddingService

def test_embeddings():
    print("Starting Embedding tests...")
    embedder = EmbeddingService()
    
    print("\n--- Testing English Model (Dense Only) ---")
    en_emb = embedder.get_en_embedding("Hello world, this is a test.")
    print(f"English dense embedding vector size: {len(en_emb)}")
    print(f"Sample values (first 5): {en_emb[:5]}")
    
    print("\n--- Testing Vietnamese Model (Dense + Sparse) ---")
    vi_text = "Xin chào thế giới, hôm nay là một ngày đẹp trời."
    vi_dense, vi_sparse = embedder.get_vi_embeddings(vi_text)
    
    print(f"Vietnamese dense embedding vector size: {len(vi_dense)}")
    print(f"Vietnamese sparse indices count: {len(vi_sparse['indices'])}")
    print(f"Sample sparse indices/values mapping: {list(zip(vi_sparse['indices'][:5], vi_sparse['values'][:5]))}")
    print("Embedding tests completed successfully!")

if __name__ == "__main__":
    test_embeddings()
