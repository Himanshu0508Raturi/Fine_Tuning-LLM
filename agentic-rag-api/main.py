from fastapi import FastAPI
from schema import QueryRequest , QueryResponse , AgentState
from agentic_rag import ask_question
from mangum import Mangum

app = FastAPI(
    title="Agentic RAG API",
    version="1.0.0"
)
handler = Mangum(app)
@app.post("/query",response_model=QueryResponse)
def query_rag(request: QueryRequest):
    state = AgentState(question=request.question)
    result = ask_question(state)
    return QueryResponse(answer=result["ans"])