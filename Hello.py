import streamlit as st
from utils import run_rag_pipeline

# Page setup
st.set_page_config(
        page_title="ESEF Reports Chat",
        page_icon="📈",
    )
st.title("ESEF Reports Chat")

# Initialize session state
if "prompt_history" not in st.session_state:
    st.session_state.prompt_history = """
    <|im_start|>user
    """
if "history" not in st.session_state:
    st.session_state.history = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show current chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Run RAG pipeline when user enters new question
if prompt := st.chat_input("Ask a question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        filter = "" #default: no filter; example: filter = "title eq 'finnair'"
        k = 2 # number of search results to be included in the prompt
        temperature = 0.2 # controls the randomness of the generation.

        # load chat history from session state for multi-question conversations
        history = st.session_state.history
        prompt_history = st.session_state.prompt_history

        answer, prompt_history, history = run_rag_pipeline(user_input=prompt, temperature=temperature, filter=filter, k=k, prompt_history=prompt_history, history=history)

        # Save chat history in session state
        st.session_state.history = history
        st.session_state.prompt_history = prompt_history

        message_placeholder.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})