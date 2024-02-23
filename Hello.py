import streamlit as st
from utils import run_rag_pipeline

# Page setup
st.set_page_config(
        page_title="ESEF Reports Chat",
        page_icon="📈",
    )
st.title("ESEF Reports - Chat")

# Initialize session state
if "prompt_history" not in st.session_state:
    st.session_state.prompt_history = """
    <|im_start|>user
    """
if "history" not in st.session_state:
    st.session_state.history = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "filter_docs" not in st.session_state:
    st.session_state.filter_docs = []

# if chat is empty, add initial welcome message by chat bot
if st.session_state.messages == []:
    st.session_state.messages.append({"role": "assistant", "content": 'Hei, olen tekoäly-chatbot, jolle on syötetty tietoja suomalaisten yritysten vuosikertomuksista. **Voit vapaasti kysyä minulta mitä** vain haluat tietää, esim. "Kuka oli FinnAirin tilintarkastaja?" tai "Millaista kestävää toimintaa Fortum harjoittaa?".'})

# Show current chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Run RAG pipeline when user enters new question
if prompt := st.chat_input("Kysy kysymys..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        # Adjust filter
        filter_docs = st.session_state.filter_docs
        if len(filter_docs) <= 0:
            filter = "title eq 'no sources provided'"
        else:
            filter = "title eq '" + filter_docs[0] + "'"
            if len(filter_docs) > 1:
                for filter_doc in filter_docs[1:]:
                    filter += " or title eq '" + filter_doc + "'"

        #st.warning(filter, icon="⚠️")

        # TODO: adjust k and temp?
        k = 5 # number of search results to be included in the prompt
        temperature = 0.2 # controls the randomness of the generation.

        # load chat history from session state for multi-question conversations
        history = st.session_state.history
        prompt_history = st.session_state.prompt_history

        answer, prompt_history, history, sources = run_rag_pipeline(user_input=prompt, temperature=temperature, filter=filter, k=k, prompt_history=prompt_history, history=history)

        # Save chat history in session state
        st.session_state.history = history
        st.session_state.prompt_history = prompt_history

        message_placeholder.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    
    # Add source information
    if len(sources) == 0:
        st.warning("No sources", icon="⚠️")
    for source in sources:
        #st.markdown(source)
        with st.expander("Lue lähde  [" + source[0] + "]..."):
                st.write(source[1])

def change_filter(data_source: str, checked: bool):
    filter_docs = st.session_state.filter_docs
    if checked:
        if data_source not in filter_docs:
            filter_docs.append(data_source)
            st.session_state.filter_docs = filter_docs
    else:
        if data_source in filter_docs:
            filter_docs.remove(data_source)
            st.session_state.filter_docs = filter_docs

with st.sidebar:
    st.title("Valitse lähdeasiakirjat: ")
    for data_source in ["finnair","yitgroup","nokia","tietoevry","citycon","743700G7A9J1PHM3X223-2022-12-31-FI","srv","fortum"," outokumpu","qt","nokianrenkaat","uponor"]:
        if st.checkbox(data_source, value=True):
            change_filter(data_source, True)
        else: 
            change_filter(data_source, False)
           # st.session_state.filter_docs.append(data_source)

    # TODO: adjust k and temp?
