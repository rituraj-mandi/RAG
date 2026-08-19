import requests
r = requests.head('https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/hinval.parquet', allow_redirects=True)
print(int(r.headers.get('content-length', 0)) / 1024 / 1024)
