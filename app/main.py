from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.services.retriever import Retriever
from app.services.llm import LLMService
from app.routes.ask import make_ask_router
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

retriever = Retriever(
    index_path="data/mqsim.index",
    metadata_path="data/mqsim_metadata.json",
    model_name="BAAI/bge-m3",
)
llm = LLMService(api_key=os.getenv("GROQ_API_KEY"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(title="MQSim Codebase RAG", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(make_ask_router(retriever, llm))
# in main.py
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.get("/health")
def health():
    return {"status": "ok", "chunks_loaded": len(retriever.metadata)}