import time
import numpy as np
from rag_pipeline import RAGPipeline

def main():
    print("Initializing RAG Pipeline for evaluation...")
    rag = RAGPipeline()
    
    test_queries = [
        "What is the capital of India?",
        "How do I brew French press coffee?",
        "What are the benefits of drinking green tea?",
        "Who wrote the play Hamlet?",
        "Explain the theory of relativity in simple terms.",
        "What is the average temperature in New York in December?",
        "How many planets are in the solar system?",
        "What are the symptoms of the common cold?",
        "How do you cook a perfect steak?",
        "What is the tallest mountain in the world?"
    ]
    
    latencies = []
    
    print(f"Running {len(test_queries)} test queries...")
    for q in test_queries:
        try:
            res = rag.run_pipeline_text(q)
            latencies.append(res['metrics']['total_ms'])
            print(f"Query: '{q[:30]}...' -> Latency: {res['metrics']['total_ms']}ms")
        except Exception as e:
            print(f"Query failed: {e}")
            
    if latencies:
        p50 = np.percentile(latencies, 50)
        p70 = np.percentile(latencies, 70)
        p100 = np.percentile(latencies, 100)
        
        print("\n=== Latency Analytics ===")
        print(f"Total Queries: {len(latencies)}")
        print(f"P50 Latency: {p50:.2f} ms")
        print(f"P70 Latency: {p70:.2f} ms")
        print(f"P100 Latency: {p100:.2f} ms")
        print("=========================")
    else:
        print("No successful queries to evaluate.")

if __name__ == "__main__":
    main()
