from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from rag_pipeline import RAGPipeline

app = FastAPI(title="Voice RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAGPipeline()

class QueryRequest(BaseModel):
    query: str

@app.post("/api/query")
async def query_text(request: QueryRequest):
    result = rag.run_pipeline_text(request.query)
    return result

@app.post("/api/voice")
async def query_voice(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    result = rag.run_pipeline_audio(audio_bytes)
    return result

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
