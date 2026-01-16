import os
from typing import TypedDict, List
from langgraph.graph import StateGraph , END
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone.grpc import PineconeGRPC as Pinecone
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from dotenv import load_dotenv
from schema import AgentState
load_dotenv()

llm = ChatGroq(
    groq_api_key=os.getenv("NEW_GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0.1,
    max_tokens=1024
)

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(host=os.getenv("PINECONE_HOST"))

# ------------------ AGENT NODES ------------------

def decide_retrieval(state: AgentState) -> AgentState:
    prompt = f"""
Decide if retrieval is needed.

User Query:
"{state.question}"

Respond with ONLY:
RETRIEVE or GENERATE
"""
    response = llm.invoke(prompt)
    decision = response.content.strip().upper()
    return state.model_copy(update={
        "needs_retrieval": decision != "GENERATE"
    })


def retrieve_document(state: AgentState) -> AgentState:
    embedding = embedder.encode(state.question).tolist()

    result = index.query(
        vector=embedding,
        top_k=5,
        namespace="rag-docs",
        include_metadata=True
    )

    docs = [m.metadata["text"] for m in result.matches]
    return state.model_copy(update={"documents": docs})


def generate_ans(state: AgentState) -> AgentState:
    if state.documents:
        context = "\n\n".join(state.documents)
        prompt = f"Context:\n{context}\n\nQuestion:\n{state.question}"
    else:
        prompt = state.question

    response = llm.invoke(prompt)
    return state.model_copy(update={"ans": response.content})


def should_retrieve(state: AgentState) -> str:
    return "retrieve" if state.needs_retrieval else "generate"

# ------------------ GRAPH ------------------

workflow = StateGraph(AgentState)

workflow.add_node("decide", decide_retrieval)
workflow.add_node("retrieve", retrieve_document)
workflow.add_node("generate", generate_ans)

workflow.set_entry_point("decide")

workflow.add_conditional_edges(
    "decide",
    should_retrieve,
    {"retrieve": "retrieve", "generate": "generate"}
)

workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()


def ask_question(state: AgentState):
    return app.invoke(state)