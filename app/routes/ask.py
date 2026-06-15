from fastapi import APIRouter, HTTPException
from app.models.schemas import AskRequest, AskResponse, SourceInfo
from app.services.retriever import Retriever
from app.services.llm import LLMService

router = APIRouter()

def build_context(chunks: list[dict]) -> str:
    sections = []
    for c in chunks:
        header = f"[{c['type'].upper()}] {c['name']}"
        if c["class_name"]:
            header += f" (class: {c['class_name']})"
        header += f"\nFile: {c['file'].split('/')[-1]}:{c['line_start']}"
        sections.append(f"{header}\n\n{c['content']}")
    return "\n\n---\n\n".join(sections)

def make_ask_router(retriever: Retriever, llm: LLMService) -> APIRouter:
    @router.post("/ask", response_model=AskResponse)
    async def ask(req: AskRequest):
        if not req.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty.")

        chunks, total = retriever.search(req.question, req.top_k)

        if not chunks:
            return AskResponse(
                answer="No confident matches found. Try MQSim-specific terms like 'GC_and_WL_Unit', 'FTL', 'NVMe submission queue'.",
                sources=[],
                chunks_searched=total,
            )

        response = llm.complete(req.question, build_context(chunks))

        return AskResponse(
            answer=response,
            sources=[
                SourceInfo(
                    name=c["name"],
                    type=c["type"],
                    file=c["file"].split("/")[-1],
                    line=c["line_start"],
                    score=c["score"],
                )
                for c in chunks
            ],
            chunks_searched=total,
        )

    return router