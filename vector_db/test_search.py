from qdrant_pipeline import QdrantPipeline

def test_search():
    print("Initializing Qdrant Pipeline...")
    pipeline = QdrantPipeline(storage_path="./qdrant_data")
    
    # 1. Indexing test English
    en_docs = [
        "The quick brown fox jumps over the lazy dog",
        "Artificial intelligence is transforming the technological landscape",
        "A simple test for vector databases using dense search"
    ]
    print("\nIndexing English documents...")
    for doc in en_docs:
        pipeline.index_document(doc, lang="en")
        
    # 2. Indexing test Vietnamese
    vi_docs = [
        "Trí tuệ nhân tạo đang thay đổi thế giới công nghệ hiện nay",
        "Con cáo màu nâu nhanh nhẹn nhảy qua một con chó lười biếng",
        "Thử nghiệm cơ sở dữ liệu vector kết hợp tìm kiếm từ khóa và ngữ nghĩa"
    ]
    print("Indexing Vietnamese documents...")
    for doc in vi_docs:
        pipeline.index_document(doc, lang="vi")
        
    print("All documents indexed successfully.")
    
    # 3. Search test English
    query_en = "fox jumping"
    print(f"\n--- English Search Test ---")
    print(f"Query: '{query_en}'")
    results_en = pipeline.search(query_en, lang="en", top_k=2)
    for r in results_en:
        print(f"  > Score: {r.score:.4f} | Text: {r.payload['text']}")
        
    # 4. Search test Vietnamese
    query_vi = "tìm kiếm vector ngữ nghĩa"
    print(f"\n--- Vietnamese Hybrid Search Test ---")
    print(f"Query: '{query_vi}'")
    results_vi = pipeline.search(query_vi, lang="vi", top_k=2)
    for r in results_vi:
        print(f"  > Score: {r.score:.4f} | Text: {r.payload['text']}")

    print("\nSearch tests completed successfully!")

if __name__ == "__main__":
    test_search()
