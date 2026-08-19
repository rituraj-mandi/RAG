import os
import time
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from fastembed import TextEmbedding
from sentence_transformers import CrossEncoder
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

class RAGPipeline:
    def __init__(self, qdrant_path="./qdrant_data", collection_name="msmarco_hybrid"):
        print("Initializing RAG Pipeline components...")
        self.client = QdrantClient(path=qdrant_path)
        self.collection_name = collection_name
        
        # Load embedding models for query
        self.dense_model = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        
        # Load Reranker
        print("Loading reranker...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
        
        # Load LLM
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            self.gemini_client = genai.Client(api_key=api_key)
        else:
            print("WARNING: GEMINI_API_KEY not set. Using mock LLM generator.")
            self.gemini_client = None
            
        # Using flash for lower latency
        self.llm_model = 'gemini-2.5-flash'
        
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY", "")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
    def transcribe_audio(self, audio_bytes):
        if not self.elevenlabs_api_key:
            return "What is the capital of India?" # Mock fallback for testing
            
        url = "https://api.elevenlabs.io/v1/speech-to-text"
        files = {'file': ('audio.wav', audio_bytes, 'audio/wav')}
        data = {'model_id': 'scribe_v2'}
        headers = {'xi-api-key': self.elevenlabs_api_key}
        
        response = requests.post(url, files=files, data=data, headers=headers)
        if response.status_code == 200:
            return response.json().get('text', '')
        else:
            print(f"ElevenLabs API Error: {response.text}")
            raise Exception("Failed to transcribe audio")

    def retrieve(self, query, top_k=20):
        # Generate query embeddings
        dense_vec = list(self.dense_model.query_embed(query))[0].tolist()
        query_res = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_vec,
            using="dense",
            limit=top_k
        )
        dense_hits = query_res.points
        
        # Combine unique documents
        unique_docs = {}
        for hit in dense_hits:
            doc_id = hit.id
            if doc_id not in unique_docs:
                unique_docs[doc_id] = hit.payload['text']
                
        return list(unique_docs.values())

    def rerank(self, query, documents, top_n=3):
        if not documents:
            return []
            
        pairs = [[query, doc] for doc in documents]
        scores = self.reranker.predict(pairs)
        
        # Sort documents by score
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in doc_score_pairs[:top_n]]

    def generate(self, query, documents):
        if not documents:
            return "I couldn't find any relevant information in the archive."
            
        context = "\n\n".join([f"[{i+1}] {doc}" for i, doc in enumerate(documents)])
        
        prompt = f"""You are a helpful voice assistant. Answer the user's query based ONLY on the provided context. 
If the context does not contain the answer, reply with "I'm sorry, I don't have information about that."
Do not hallucinate or use outside knowledge. Be concise.

Context:
{context}

Query: {query}
Answer:"""

        if not self.gemini_client:
            time.sleep(0.5)
            return "Mock answer since API key is absent."

        response = self.gemini_client.models.generate_content(
            model=self.llm_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=150
            )
        )
        return response.text.strip()

    def run_pipeline_text(self, query):
        metrics = {}
        
        t0 = time.time()
        docs = self.retrieve(query, top_k=20)
        metrics['retrieval_ms'] = int((time.time() - t0) * 1000)
        
        t1 = time.time()
        top_docs = self.rerank(query, docs, top_n=3)
        metrics['rerank_ms'] = int((time.time() - t1) * 1000)
        
        t2 = time.time()
        answer = self.generate(query, top_docs)
        metrics['generation_ms'] = int((time.time() - t2) * 1000)
        
        metrics['total_ms'] = metrics['retrieval_ms'] + metrics['rerank_ms'] + metrics['generation_ms']
        
        return {
            "query": query,
            "answer": answer,
            "context": top_docs,
            "metrics": metrics
        }
        
    def run_pipeline_audio(self, audio_bytes):
        t_stt = time.time()
        query = self.transcribe_audio(audio_bytes)
        stt_ms = int((time.time() - t_stt) * 1000)
        
        res = self.run_pipeline_text(query)
        res['metrics']['stt_ms'] = stt_ms
        res['metrics']['total_ms'] += stt_ms
        return res
