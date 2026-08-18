from datasets import load_dataset
import json

def fetch_sample():
    print("Loading MSMARCO-XI...")
    ds = load_dataset("ai4bharat/MSMARCO-XI", "default", split="train", streaming=True)
    
    samples = []
    for idx, item in enumerate(ds):
        print(item.keys())
        samples.append(item)
        if idx >= 200:
            break
            
    with open("sample.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print("Saved 200 samples to sample.json")

if __name__ == "__main__":
    fetch_sample()
