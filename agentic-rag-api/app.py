import streamlit as st
from schema import QueryRequest , QueryResponse , AgentState
from agentic_rag import ask_question

question = st.text_input("Enter your question:")

if question:
    initial_state = AgentState(question=question)
    final_state = ask_question(initial_state)
    st.write("Answer:", final_state["ans"])