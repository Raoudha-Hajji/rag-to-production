"""
Demo interface for the arXiv agent.

> streamlit run src/app.py
"""

import streamlit as st
from agent import run_agent

st.set_page_config(page_title="ArXiv Research Agent", page_icon="🔎", layout="centered")

st.title("🔎 ArXiv Research Agent")
st.caption(
    "An agentic RAG assistant that answers questions about AI/NLP research. "
    "It searches a local indexed paper collection first, and falls back to "
    "live arXiv search when needed."
)

# st.session_state persists values across reruns — Streamlit reruns the
# whole script on every interaction, so without this, chat history would
# reset every time you ask a new question.
if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("Ask a question about AI/NLP research:")

col1, col2 = st.columns([1, 5])
with col1:
    ask_clicked = st.button("Ask", type="primary")

if ask_clicked and question:
    with st.spinner("Thinking... (the agent may search papers, this can take a few seconds)"):
        answer = run_agent(question, verbose=False)
    st.session_state.history.insert(0, {"question": question, "answer": answer})

# Display conversation history, most recent first
for entry in st.session_state.history:
    st.markdown(f"**Q: {entry['question']}**")
    st.markdown(entry["answer"])
    st.divider()