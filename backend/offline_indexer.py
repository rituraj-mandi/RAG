import os
import uuid
from datasets import load_dataset
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from fastembed import TextEmbedding, SparseTextEmbedding
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')


def get_dir_size(path="."):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total

def main():
    import argparse
    import time
    from dotenv import load_dotenv
    
    parser = argparse.ArgumentParser(description="Index MSMARCO dataset into Qdrant")
    parser.add_argument("--reset", action="store_true", help="Delete and recreate the Qdrant collection")
    args = parser.parse_args()
    
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    
    print("Loading dataset from HuggingFace...")
    if not hf_token:
        print("WARNING: HF_TOKEN not found in environment. Streaming might be rate-limited or blocked.")
        
    try:
        # Load dataset in streaming mode so we don't download everything into memory
        ds = load_dataset('ai4bharat/MSMARCO-XI', 'default', split='train', streaming=True, token=hf_token)
    except Exception as e:
        print(f"Failed to connect to HuggingFace: {e}")
        return

    client = QdrantClient(path="./qdrant_data")
    collection_name = "msmarco_hybrid"
    
    if args.reset:
        print(f"Reset flag passed. Deleting existing collection '{collection_name}' if it exists...")
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
    
    print("Loading embedding models...")
    dense_model = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={"dense": VectorParams(size=384, distance=Distance.COSINE)}
        )
        print("Created Qdrant collection.")
    else:
        print("Collection already exists, appending new points.")

    max_items = 50000
    batch_size = 64
    docs_to_embed = []
    metadata_list = []
    
    count = 0
    total_indexed = 0
    start_time = time.time()
    
    print(f"Streaming and indexing up to {max_items} passages...")
    for item in ds:
        if count >= max_items:
            break
            
        text = item.get("english_passage", "")
        hindi = item.get("hindi_passage", "")
        if not text:
            continue
            
        full_text = f"{text}\n{hindi}"
        
        # 1 valid passage -> 1 embedding -> 1 Qdrant point
        docs_to_embed.append(full_text)
        metadata_list.append({"text": full_text, "strategy": "passage", "doc_id": str(count)})
        
        count += 1
        
        # Process in batches to save memory
        if len(docs_to_embed) >= batch_size:
            dense_embeds = list(dense_model.embed(docs_to_embed))
            batch_points = []
            for j, (doc, meta) in enumerate(zip(docs_to_embed, metadata_list)):
                point_id = total_indexed + j + 1
                point = PointStruct(
                    id=point_id,
                    vector={"dense": dense_embeds[j].tolist()},
                    payload=meta
                )
                batch_points.append(point)
                
            client.upsert(collection_name=collection_name, points=batch_points)
            total_indexed += len(docs_to_embed)
            docs_to_embed = []
            metadata_list = []
            
            if count % 1000 == 0:
                print(f"Progress: Read {count}/{max_items} passages. Indexed {total_indexed} chunks.")

    # Process any remaining documents
    if docs_to_embed:
        dense_embeds = list(dense_model.embed(docs_to_embed))
        batch_points = []
        for j, (doc, meta) in enumerate(zip(docs_to_embed, metadata_list)):
            point_id = total_indexed + j + 1
            point = PointStruct(
                id=point_id,
                vector={"dense": dense_embeds[j].tolist()},
                payload=meta
            )
            batch_points.append(point)
            
        client.upsert(collection_name=collection_name, points=batch_points)
        total_indexed += len(docs_to_embed)

    elapsed_time = time.time() - start_time
    passages_per_sec = count / elapsed_time if elapsed_time > 0 else 0
    
    try:
        db_size_bytes = get_dir_size("./qdrant_data")
        db_size_mb = db_size_bytes / (1024 * 1024)
    except Exception:
        db_size_mb = 0

    print("\n" + "="*40)
    print("INDEXING SUMMARY")
    print("="*40)
    print(f"Passages processed:   {count}")
    print(f"Vectors indexed:      {total_indexed}")
    print(f"Elapsed time:         {elapsed_time:.2f} seconds")
    print(f"Passages/sec:         {passages_per_sec:.2f}")
    print(f"Qdrant database size: {db_size_mb:.2f} MB")
    print("="*40)


if __name__ == "__main__":
    main()
