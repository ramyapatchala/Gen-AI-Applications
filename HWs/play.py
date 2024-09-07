import time
import requests
import fitz  # PyMuPDF for reading PDFs
import cohere
import streamlit as st

# Load Cohere API key from environment variables
cohere_key = st.secrets.get('cohere_key')

# Function to read PDF content from a URL using PyMuPDF (fitz).
def read_pdf_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with fitz.open(stream=response.content, filetype="pdf") as doc:
            document = ""
            for page in doc:
                document += page.get_text()
        return document
    except requests.RequestException as e:
        st.error(f"Error fetching PDF from {url}: {e}")
        return None
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None

# Streamlit app title and description.
st.title("📄 PDF Summarizer from URL")
st.write("Enter a PDF URL and select summary options below.")

# Function to validate Cohere API key.
def verify_cohere_key(api_key):
    try:
        client = cohere.Client(api_key)
        # Try a simple API call to validate the key
        client.generate(prompt="Hello", max_tokens=5)
        return client, True, "API key is valid"
    except Exception as e:
        return None, False, str(e)

# Validate the Cohere API key
client, is_valid, message = verify_cohere_key(cohere_key)

if is_valid:
    st.sidebar.success("Cohere API key is valid!", icon="✅")
else:
    st.sidebar.error(f"Invalid Cohere API key: {message}", icon="❌")
    st.stop()

# Sidebar: Provide the user with summary options.
st.sidebar.header("Summary Options")
summary_option = st.sidebar.radio(
    "Choose how you would like the document to be summarized:",
    options=[
        "Summarize the document in 100 words",
        "Summarize the document in 2 connecting paragraphs",
        "Summarize the document in 5 bullet points"
    ]
)

# Sidebar: Provide a dropdown menu for language selection
language_option = st.sidebar.selectbox(
    "Choose the output language:",
    options=["English", "French", "Spanish"]
)

# Let the user enter a URL for the PDF
url = st.text_input("Enter the URL to the PDF:")

# Ensure the `url` variable is properly initialized
if url:
    # Initialize document variable
    document = read_pdf_from_url(url)

    # If document is successfully loaded from URL
    if document:
        # Modify the prompt based on the selected summary option.
        if summary_option == "Summarize the document in 100 words":
            summary_instruction = "Summarize this document in 100 words."
        elif summary_option == "Summarize the document in 2 connecting paragraphs":
            summary_instruction = "Summarize this document in 2 connecting paragraphs."
        else:
            summary_instruction = "Summarize this document in 5 bullet points."

        # Adjust the prompt to include the chosen language
        if language_option == "English":
            language_instruction = "Please summarize the document in English."
        elif language_option == "Spanish":
            language_instruction = "Por favor, resuma el documento en español."
        else:
            language_instruction = "Veuillez résumer le document en français."

        # Combine document, summary, and language instructions.
        prompt = f"{summary_instruction} {language_instruction} \n\n\n---\n\n Document: {document}\n\n---\n\n"
        try:
        # Generate summary using Cohere
            events = client.chat_stream(
                        model='command-r',
                        message=prompt,
                        temperature=0,       
                        max_tokens=1500,
                        prompt_truncation='AUTO',
                        connectors=[],
                        documents=[]
            )
            st.write(events)
            response_text=""
            for event in events:
                if event.event_type=="text-generation":
                    response_text = response_text + str(event.text)
            st.write(response_text)

        except Exception as e:
            st.error(f"Unexpected error: {e}", icon="❌")
else:
    st.info("Please enter a valid PDF URL to generate a summary.", icon="🌐")
