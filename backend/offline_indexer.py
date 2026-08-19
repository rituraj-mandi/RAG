import os
import uuid
from dotenv import load_dotenv
from datasets import load_dataset

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is not set")
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, SparseVectorParams, SparseIndexParams, SparseVector
from fastembed import TextEmbedding, SparseTextEmbedding
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')

def get_sliding_window_chunks(text, window_size=300, overlap=50):
    words = text.split()
    if len(words) <= window_size:
        return [text]
    
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+window_size])
        chunks.append(chunk)
        i += (window_size - overlap)
    return chunks

def get_semantic_chunks(text, max_words=100):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        words = len(sentence.split())
        if current_length + words > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence)
        current_length += words
        
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def main():
    print("Loading MSMARCO-XI streaming dataset...")
    # Stream the dataset to avoid large memory/disk footprint and rate limits
    ds = load_dataset(
        "ai4bharat/MSMARCO-XI",
        "default",
        split="train",
        streaming=True,
        token=HF_TOKEN
    )
    
    unique_passages = {}
    max_items = 500  # Adjust as needed for deployment size
    
    for idx, item in enumerate(ds):
        if idx >= max_items:
            break
            
        # MSMARCO-XI schema has passages['English_passages'] list
        eng_passages = item.get("passages", {}).get("English_passages", [])
        
        # Take the first passage for this query's context
        if eng_passages:
            text = eng_passages[0]
            if text not in unique_passages:
                unique_passages[text] = {"source": "msmarco", "doc_id": str(uuid.uuid4())[:8]}
                
    print(f"Extracted {len(unique_passages)} unique passages from stream.")
    
    client = QdrantClient(path="./qdrant_data")
    collection_name = "msmarco_hybrid"
    
    print("Loading embedding models (Dense and Sparse)...")
    dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={"dense": VectorParams(size=384, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))}
        )
        print("Created Qdrant collection with hybrid capabilities.")
    else:
        print("Collection already exists, appending points.")
    
    docs_to_embed = []
    metadata_list = []
    
    for text, meta in unique_passages.items():
        # Strategy A: Passage
        docs_to_embed.append(text)
        metadata_list.append({"text": text, "strategy": "passage", "doc_id": meta["doc_id"]})
        
        # Strategy B: Semantic
        sem_chunks = get_semantic_chunks(text)
        if len(sem_chunks) > 1:
            for sc in sem_chunks:
                docs_to_embed.append(sc)
                metadata_list.append({"text": sc, "strategy": "semantic", "doc_id": meta["doc_id"]})
        
        # Strategy C: Sliding Window
        sw_chunks = get_sliding_window_chunks(text, window_size=50, overlap=10)
        if len(sw_chunks) > 1:
            for swc in sw_chunks:
                docs_to_embed.append(swc)
                metadata_list.append({"text": swc, "strategy": "sliding_window", "doc_id": meta["doc_id"]})
                
    print(f"Total chunks to index: {len(docs_to_embed)}")
    
    batch_size = 64
    for i in range(0, len(docs_to_embed), batch_size):
        batch_docs = docs_to_embed[i:i+batch_size]
        batch_meta = metadata_list[i:i+batch_size]
        
        dense_embeds = list(dense_model.embed(batch_docs))
        sparse_embeds = list(sparse_model.embed(batch_docs))
        
        batch_points = []
        for j, (doc, meta) in enumerate(zip(batch_docs, batch_meta)):
            dense_vec = dense_embeds[j].tolist()
            sparse_vec = SparseVector(
                indices=sparse_embeds[j].indices.tolist(),
                values=sparse_embeds[j].values.tolist()
            )
            
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_vec,
                    "sparse": sparse_vec
                },
                payload=meta
            )
            batch_points.append(point)
            
        client.upsert(collection_name=collection_name, points=batch_points)
        print(f"Indexed batch {i//batch_size + 1}/{(len(docs_to_embed)+batch_size-1)//batch_size}")
        
    print("Indexing complete!")

if __name__ == "__main__":
    main()
