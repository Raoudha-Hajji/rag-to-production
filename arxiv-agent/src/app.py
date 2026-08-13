"""
Demo interface for the arXiv agent.

Run:
    streamlit run src/app.py
"""

import streamlit as st
from agent import run_agent


st.set_page_config(
    page_title="ArXiv Research Agent",
    page_icon="🔎",
    layout="wide",
)


if "history" not in st.session_state:
    st.session_state.history = []


with st.sidebar:
    st.header("🔎 ArXiv Research Agent")

    st.markdown(
        """
        An agentic RAG assistant for AI/NLP research.

        **How it works**
        - Searches the local paper collection
        - Uses semantic retrieval
        - Falls back to live arXiv search
        - Generates an answer based on retrieved papers
        """
    )

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()


st.title("🔎 ArXiv Research Agent")

st.caption(
    "Ask questions about AI and NLP research. "
    "The agent searches your indexed papers first and "
    "uses live arXiv search when needed."
)


for entry in st.session_state.history:

    with st.chat_message("user", avatar="👤"):
        st.write(entry["question"])

    with st.chat_message("assistant", avatar="🔎"):

        st.markdown(entry["answer"])

        if entry.get("sources"):
            with st.expander(
                f"📚 Sources ({len(entry['sources'])})"
            ):
                for i, source in enumerate(entry["sources"], 1):
                    st.markdown(
                        f"**[{i}] [{source['title']}]({source['url']})**  \n"
                        f"<small>{source.get('authors', 'Unknown authors')}</small>",
                        unsafe_allow_html=True,
                    )


question = st.chat_input(
    "Ask a question about AI/NLP research..."
)


if question:

    with st.chat_message("user", avatar="👤"):
        st.write(question)

    with st.chat_message("assistant", avatar="🔎"):

        with st.spinner("Searching papers and generating an answer..."):
            answer, sources = run_agent(
                question,
                verbose=False,
            )

        st.markdown(answer)

        if sources:
            with st.expander(
                f"📚 Sources ({len(sources)})"
            ):
                for i, source in enumerate(sources, 1):
                    st.markdown(
                        f"**[{i}] [{source['title']}]({source['url']})**  \n"
                        f"<small>{source.get('authors', 'Unknown authors')}</small>",
                        unsafe_allow_html=True,
                    )

    st.session_state.history.append(
        {
            "question": question,
            "answer": answer,
            "sources": sources,
        }
    )