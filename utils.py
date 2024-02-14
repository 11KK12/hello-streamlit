# Imports
from azure.core.credentials import AzureKeyCredential
import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt
from azure.search.documents import SearchClient
from azure.search.documents.models import Vector 

import streamlit as st

# Set up openai
openai.api_type = st.secrets["OPENAI_API_TYPE"]
openai.api_key = st.secrets["OPENAI_API_KEY"]
openai.api_base = st.secrets["OPENAI_API_BASE"]
openai.api_version = st.secrets["OPENAI_API_VERSION"]
gpt_engine_name = st.secrets["GPT_ENGINE_NAME"]
embedding_engine_name = st.secrets["EMBEDDING_ENGINE_NAME"]
search_index_name = "esef_reports_rag_index"
search_endpoint = st.secrets["SEARCH_ENDPOINT"]
search_key= st.secrets["SEARCH_KEY"]

summary_prompt_template = """Alla on yhteenveto tähänastisesta keskustelusta ja käyttäjän esittämä uusi kysymys, johon on vastattava etsimällä tietopankista. Luo hakukysely keskustelun ja uuden kysymyksen perusteella. Lähteiden nimet eivät ole hyviä hakusanoja, jotka kannattaa sisällyttää hakukyselyyn.

Yhteenveto:
{summary}

Kysymys:
{question}

Hakukysely:
"""

# Prompt Template Finnish
prompt_prefix = """<|im_start|>system
Assistentti auttaa rahoitusvalvojia löytämään tietoja rahoituslaitoksista. 
Vastaa AINOASTAAN alla olevassa lähdeluettelossa luetelluilla tosiasioilla. Jos alla olevat tiedot eivät riitä, sano, ettet tiedä. Älä luo vastauksia, joissa ei käytetä alla olevia lähteitä. Jos selventävän kysymyksen esittäminen käyttäjälle auttaisi, esitä kysymys. 
Jokaisessa lähteessä on nimi, jota seuraa kaksoispiste ja varsinainen tieto. Ilmoita aina lähteen nimi jokaisesta vastauksessa käyttämästäsi tiedosta. Käytä lähdeviittaukseen neliöjalkoja, esim. [info_1]. Älä yhdistä lähteitä, vaan mainitse jokainen lähde erikseen, esim. [info_1][info_2].

Lähteet:
{sources}

<|im_end|>"""

# RAG Preparation
turn_prefix = """
<|im_start|>user
"""
turn_suffix = """
<|im_end|>
<|im_start|>assistant
"""
prompt_history = turn_prefix
history = []

# Functions
def search(query: str, filter="", k=5):
    credential = AzureKeyCredential(search_key)
    search_client = SearchClient(endpoint=search_endpoint, index_name=search_index_name, credential=credential)

    vector = Vector(value=generate_embeddings(query, embedding_engine_name), k=3, fields="contentVector")

    results = search_client.search(  
        search_text=query,
        vectors= [vector],
        filter=filter,
        select=["id", "title", "content"],
        top=k
    ) 

    return results

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
# Function to generate embeddings for title and content fields, also used for query embeddings
def generate_embeddings(text: str, embedding_engine_name: str) -> list:
    response = openai.Embedding.create(
        input=text, engine=embedding_engine_name)
    embeddings = response['data'][0]['embedding'] # output dimensions = 1536
    return embeddings

def run_rag_pipeline(user_input: str, temperature: int, filter: str, k: int, prompt_history: str, history: list):

    # Generating search query based on user input and chat history
    if len(history) > 0:
        completion = openai.Completion.create(
            engine=gpt_engine_name,
            prompt=summary_prompt_template.format(summary="\n".join(history), question=user_input),
            temperature=temperature,
            max_tokens=32,
            stop=["\n"])
        query = completion.choices[0].text
    else:
        query = user_input

    # Searching for documents
    r = search(query=query, filter=filter, k=k)
    results = [doc["id"] + ": " + doc["content"].replace("\n", " ").replace("\r", " ") for doc in r]
    content = "\n".join(results)

    # Create prompt with content and context
    prompt = prompt_prefix.format(sources=content) + prompt_history + user_input + turn_suffix

    # Run prompt
    completion = openai.Completion.create(
        engine=gpt_engine_name, 
        prompt=prompt, 
        temperature=temperature, 
        max_tokens=1024,
        stop=["<|im_end|>", "<|im_start|>"])

    # Update history
    answer = completion.choices[0].text
    prompt_history += user_input + turn_suffix + answer + "\n<|im_end|>" + turn_prefix
    history.append("user: " + user_input)
    history.append("assistant: " + completion.choices[0].text)

    return answer, prompt_history, history
