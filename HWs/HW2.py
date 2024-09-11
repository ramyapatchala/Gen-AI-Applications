import streamlit as st
import requests
from bs4 import BeautifulSoup  # For extracting webpage content
from openai import OpenAI, OpenAIError
import cohere
from mistralai import Mistral

# Function to read webpage content from a URL
def read_webpage_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        document = " ".join([p.get_text() for p in soup.find_all("p")])
        return document
    except requests.RequestException as e:
        st.error(f"Error reading webpage from {url}: {e}")
        return None
    except Exception as e:
        st.error(f"Error processing the webpage: {e}")
        return None

# Function to verify OpenAI API key
def verify_openai_key(api_key):
    try:
        client = OpenAI(api_key=api_key)
        client.models.list()
        return client, True, "API key is valid"
    except OpenAIError as e:
        return None, False, str(e)

# Function to verify Cohere API key
def verify_cohere_key(api_key):
    try:
        client = cohere.Client(api_key)
        client.generate(prompt="Hello", max_tokens=5)
        return client, True, "API key is valid"
    except Exception as e:
        return None, False, str(e)

# Function to verify Mistral API key
def verify_mistral_key(api_key):
    client = Mistral(api_key=api_key)
    try:
        response = client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": "Test message"}]
        )
        if response.choices:
            return client, True, "API key is valid"
        else:
            return None, False, "API key is invalid or no response received"
    except Exception as e:
        return None, False, str(e)

# Function to generate summary using OpenAI
def generate_openai_summary(client, document, summary_instruction, language_instruction, use_advanced_model):
    model_choice = "gpt-4o" if use_advanced_model else "gpt-4o-mini"
    messages = [
        {
            "role": "user",
            "content": f"Here's a document: {document} \n\n---\n\n {summary_instruction} {language_instruction}",
        }
    ]
    try:
        stream = client.chat.completions.create(
            model=model_choice,
            messages=messages,
            stream=True,
        )
        return stream
    except OpenAIError as e:
        st.error(f"Error generating summary: {e}", icon="❌")
        return None

# Function to generate summary using Cohere
def generate_cohere_summary(client, document, summary_instruction, language_instruction):
    prompt = f"{language_instruction} {summary_instruction} \n\n\n---\n\n Document: {document}\n\n---\n\n"
    try:
        events = client.chat_stream(
            model='command-r',
            message=prompt,
            temperature=0,       
            max_tokens=1500,
            prompt_truncation='AUTO',
            connectors=[],
            documents=[]
        )
        return events
    except Exception as e:
        st.error(f"Error generating summary: {e}", icon="❌")
        return None

# Function to generate summary using Mistral
def generate_mistral_summary(client, document, summary_instruction, language_instruction):
    prompt = f"{language_instruction} {summary_instruction} \n\n\n---\n\n Document: {document}\n\n---\n\n"
    try:
        chat_response = client.chat.complete(
            model="mistral-large-latest",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
        )
        return chat_response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating summary: {e}", icon="❌")
        return None

# Streamlit app
st.title("📄 Multi-LLM Webpage Summarizer from URL")
st.write("Enter a webpage URL, select your LLM provider, and choose summary options.")

# Sidebar: LLM provider selection
st.sidebar.header("LLM Provider")
llm_provider = st.sidebar.radio(
    "Choose your LLM provider:",
    options=["OpenAI", "Cohere", "Mistral"]
)

# API key verification
if llm_provider == "OpenAI":
    openai_api_key = st.secrets['key1']
    client, is_valid, message = verify_openai_key(openai_api_key)
elif llm_provider == "Cohere":
    cohere_api_key = st.secrets['cohere_key']
    client, is_valid, message = verify_cohere_key(cohere_api_key)
else:  # Mistral
    mistral_api_key = st.secrets['mistral_key']
    client, is_valid, message = verify_mistral_key(mistral_api_key)

if is_valid:
    st.sidebar.success(f"{llm_provider} API key is valid!", icon="✅")
else:
    st.sidebar.error(f"Invalid {llm_provider} API key: {message}", icon="❌")
    st.stop()

# Sidebar: Summary options
st.sidebar.header("Summary Options")
summary_option = st.sidebar.radio(
    "Choose how you would like the document to be summarized:",
    options=[
        "Summarize the document in 100 words",
        "Summarize the document in 2 connecting paragraphs",
        "Summarize the document in 5 bullet points"
    ]
)

# Language selection
language_option = st.sidebar.selectbox(
    "Choose the output language:",
    options=["English", "French", "Spanish"]
)

# OpenAI-specific option
use_advanced_model = False
if llm_provider == "OpenAI":
    use_advanced_model = st.sidebar.checkbox("Use Advanced Model (GPT-4O)")

# Webpage URL input
url = st.text_input("Enter the URL to the webpage:")

if url:
    document = read_webpage_from_url(url)
    if document:
        # Prepare summary and language instructions
        summary_instruction = summary_option.replace("Summarize the document", "Summarize this document")
        language_instruction = {
            "English": "Please summarize the document in English.",
            "French": "Veuillez résumer le document en français.",
            "Spanish": "Por favor, resuma el documento en español."
        }[language_option]

        # Generate summary based on selected LLM provider
        if llm_provider == "OpenAI":
            stream = generate_openai_summary(client, document, summary_instruction, language_instruction, use_advanced_model)
            if stream:
                st.write_stream(stream)
        elif llm_provider == "Cohere":
            events = generate_cohere_summary(client, document, summary_instruction, language_instruction)
            if events:
                response_text = ""
                for event in events:
                    if event.event_type == "text-generation":
                        response_text += str(event.text)
                st.write(response_text)
        else:  # Mistral
            summary = generate_mistral_summary(client, document, summary_instruction, language_instruction)
            if summary:
                st.write(summary)
else:
    st.info("Please enter a valid webpage URL to generate a summary.", icon="🌐")
