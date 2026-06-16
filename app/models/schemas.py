from pydantic import BaseModel

class AskRequest(BaseModel):
    #what the user sends
    question: str
    top_k: int = 8

class SourceInfo(BaseModel):
    #citation attached to chunk
    name: str
    type: str
    file: str
    line: int
    score: float

class AskResponse(BaseModel):
    #response received
    answer: str
    sources: list[SourceInfo]
    chunks_searched: int