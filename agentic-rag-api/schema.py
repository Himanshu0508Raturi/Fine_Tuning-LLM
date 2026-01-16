from pydantic import BaseModel, Field
from typing import List

class AgentState(BaseModel):
    question: str = Field(...,description="User Query")
    documents: List[str] = Field(default_factory=list)
    ans: str=Field(default="")
    needs_retrieval: bool = Field(default=True)

class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str