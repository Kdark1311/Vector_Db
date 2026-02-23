import uuid
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, PointStruct, SparseVector
from embedding_service import EmbeddingService

class QdrantPipeline:
    def __init__(self, collection_name="hybrid_collection", storage_path="./qdrant_data"):
        # Initialize embedded local Qdrant instance.
        self.client = QdrantClient(path=storage_path) 
        self.collection_name = collection_name
        
        # Load embedding models
        self.embedder = EmbeddingService()
        self.init_collection()
        
    def init_collection(self):
        """Create the collection with dense and sparse vector configurations."""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense-en": VectorParams(
                        size=384, 
                        distance=Distance.COSINE,
                        on_disk=True  # Lưu vector trực tiếp lên ổ cứng thay vì RAM
                    ),
                    "dense-vi": VectorParams(
                        size=1024, 
                        distance=Distance.COSINE,
                        on_disk=True  # Bật on_disk=True cho vector lớn của bge-m3
                    )
                },
                sparse_vectors_config={
                    "sparse-vi": SparseVectorParams()
                },
                # Cấu hình Scalar Quantization INT8 giúp giảm dung lượng RAM xuống ~4 lần
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        always_ram=True,
                        quantile=0.99
                    )
                ),
                # Tối ưu đồ thị HNSW để đạt tốc độ tìm kiếm dưới mili giây với độ chính xác cao
                hnsw_config=models.HnswConfigDiff(
                    m=16,
                    ef_construct=100
                ),
                # Bật on_disk_payload để tiết kiệm RAM với khối lượng metadata lớn
                on_disk_payload=True
            )
            print(f"Collection '{self.collection_name}' created successfully with Production configs.")
            
    def index_document(self, text: str, lang: str = "en", metadata: dict = None):
        """Index a document into Qdrant using the correct embeddings based on the language."""
        if metadata is None:
            metadata = {}
        metadata['text'] = text
        metadata['lang'] = lang
        
        point_id = str(uuid.uuid4())
        vector_dict = {}
        
        if lang == "en":
            dense_en = self.embedder.get_en_embedding(text)
            vector_dict["dense-en"] = dense_en
        elif lang == "vi":
            dense_vi, sparse_vi = self.embedder.get_vi_embeddings(text)
            vector_dict["dense-vi"] = dense_vi
            vector_dict["sparse-vi"] = SparseVector(
                indices=sparse_vi["indices"], 
                values=sparse_vi["values"]
            )
        else:
            raise ValueError(f"Unsupported language '{lang}'. Use 'en' or 'vi'.")
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector_dict,
                    payload=metadata
                )
            ]
        )
        return point_id
        
    def search(self, query: str, lang: str = "en", top_k: int = 5):
        """Search Qdrant collection using standard dense search or hybrid search depending on language."""
        if lang == "en":
            # Standard Dense search for English
            query_vector = self.embedder.get_en_embedding(query)
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=("dense-en", query_vector),
                limit=top_k
            )
            return results
            
        elif lang == "vi":
            # Hybrid search using new Query Points API + RRF Fusion
            dense_vi, sparse_vi = self.embedder.get_vi_embeddings(query)
            
            # Using Qdrant >= 1.7 True Hybrid Search
            results = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(
                        query=dense_vi,
                        using="dense-vi",
                        limit=top_k * 2,
                    ),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=sparse_vi["indices"],
                            values=sparse_vi["values"]
                        ),
                        using="sparse-vi",
                        limit=top_k * 2,
                    )
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k
            )
            return results.points
        else:
            raise ValueError(f"Unsupported language '{lang}'.")
