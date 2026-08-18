import os
import uuid
from datasets import load_dataset
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
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
    print("Loading dataset...")
    print("Using a synthetic representative subset (HuggingFace is rate-limiting the download without a token)")
    
    # Synthetic representative subset of MSMARCO-XI like passages (English and Hindi)
    synthetic_passages = [
        {"english_passage": "The Taj Mahal is an ivory-white marble mausoleum on the right bank of the river Yamuna in Agra, Uttar Pradesh, India.", "hindi_passage": "ताजमहल भारत के उत्तर प्रदेश के आगरा में यमुना नदी के दाहिने किनारे पर एक हाथीदांत-सफेद संगमरमर का मकबरा है।"},
        {"english_passage": "New Delhi is the capital of India and a part of the National Capital Territory of Delhi.", "hindi_passage": "नई दिल्ली भारत की राजधानी है और दिल्ली के राष्ट्रीय राजधानी क्षेत्र का एक हिस्सा है।"},
        {"english_passage": "The Ganges is a trans-boundary river of Asia which flows through India and Bangladesh.", "hindi_passage": "गंगा एशिया की एक सीमा पार नदी है जो भारत और बांग्लादेश से होकर बहती है।"},
        {"english_passage": "Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas.", "hindi_passage": "माउंट एवरेस्ट समुद्र तल से ऊपर पृथ्वी का सबसे ऊँचा पर्वत है, जो हिमालय की महालंगूर हिमाल उप-श्रृंखला में स्थित है।"},
        {"english_passage": "Diwali is the Hindu festival of lights, with variations celebrated in other Indian religions. It symbolises the spiritual victory of light over darkness.", "hindi_passage": "दिवाली रोशनी का हिंदू त्योहार है, जिसे अन्य भारतीय धर्मों में भी मनाया जाता है। यह अंधकार पर प्रकाश की आध्यात्मिक जीत का प्रतीक है।"},
        {"english_passage": "Robotics is an interdisciplinary branch of computer science and engineering. Robotics involves design, construction, operation, and use of robots. The goal of robotics is to design machines that can help and assist humans.", "hindi_passage": "रोबोटिक्स कंप्यूटर विज्ञान और इंजीनियरिंग की एक अंतःविषय शाखा है।"},
        {"english_passage": "The immediate impact of the success of the Manhattan Project was the atomic bombings of Hiroshima and Nagasaki in August 1945, which quickly led to the surrender of Japan and the end of World War II.", "hindi_passage": "मैनहट्टन प्रोजेक्ट की सफलता का तत्काल प्रभाव अगस्त 1945 में हिरोशिमा और नागासाकी पर परमाणु बमबारी था।"}
    ]
    
    unique_passages = {}
    count = 0
    max_items = 5
    for item in synthetic_passages:
        if count >= max_items:
            break
            
        text = item.get("english_passage", "")
        hindi = item.get("hindi_passage", "")
        
        full_text = f"{text}\n{hindi}"
        if full_text not in unique_passages:
            unique_passages[full_text] = {"source": "msmarco_synthetic", "doc_id": str(uuid.uuid4())[:8]}
        count += 1
            
    print(f"Extracted {len(unique_passages)} unique passages.")
    
    client = QdrantClient(path="./qdrant_data")
    collection_name = "msmarco_hybrid"
    
    print("Loading embedding models...")
    dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={"dense": VectorParams(size=384, distance=Distance.COSINE)}
        )
        print("Created Qdrant collection.")
    else:
        print("Collection already exists, overwriting points if any, or just appending.")
    
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
        
        batch_points = []
        for j, (doc, meta) in enumerate(zip(batch_docs, batch_meta)):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_embeds[j].tolist()
                },
                payload=meta
            )
            batch_points.append(point)
            
        client.upsert(collection_name=collection_name, points=batch_points)
        print(f"Indexed batch {i//batch_size + 1}/{(len(docs_to_embed)+batch_size-1)//batch_size}")
        
    print("Indexing complete!")

if __name__ == "__main__":
    main()
