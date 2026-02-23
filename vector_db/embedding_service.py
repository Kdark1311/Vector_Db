import hashlib
from sentence_transformers import SentenceTransformer
from FlagEmbedding import BGEM3FlagModel

def _token_to_id(token: str) -> int:
    """Hash a string token into a 31-bit positive integer for Qdrant sparse vector indices."""
    return int(hashlib.md5(str(token).encode('utf-8')).hexdigest(), 16) % (2**31 - 1)

class EmbeddingService:
    def __init__(self):
        print("Loading English Dense Model (all-MiniLM-L6-v2)...")
        self.en_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        print("Loading Vietnamese/Multilingual Model (bge-m3)...")
        # BGEM3FlagModel supports dense, sparse and colbert out of the box
        self.vi_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        print("Models loaded successfully.")
        
    def get_en_embedding(self, text: str):
        """Returns dense embeddings for English using all-MiniLM-L6-v2"""
        return self.en_model.encode(text).tolist()
        
    def get_vi_embeddings(self, text: str):
        """Returns both dense and sparse embeddings for Vietnamese using bge-m3"""
        output = self.vi_model.encode(
            [text], 
            return_dense=True, 
            return_sparse=True, 
            return_colbert_vecs=False
        )
        
        dense = output['dense_vecs'][0].tolist()
        sparse_dict = output['lexical_weights'][0]
        
        # Qdrant sparse vectors require integer indices and corresponding float values.
        # Handle potential hash collisions by accumulating duplicate values.
        sparse_res = {}
        for k, v in sparse_dict.items():
            idx = _token_to_id(k)
            sparse_res[idx] = sparse_res.get(idx, 0.0) + float(v)
            
        indices = list(sparse_res.keys())
        values = list(sparse_res.values())
        
        return dense, {"indices": indices, "values": values}
