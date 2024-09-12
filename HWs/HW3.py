import streamlit as st
import cohere

# Title of the app
st.title("My lab3 Question Answering Chatbot")

# Initialize the Cohere client using the API key
if 'client' not in st.session_state:
    api_key = st.secrets['cohere_key']
    st.session_state.client = cohere.Client(api_key)

# Initialize the conversation messages
if 'messages' not in st.session_state:
    st.session_state['messages'] = [{"role": "assistant", "content": "How can I help you?"}]

# Display the conversation history
for msg in st.session_state.messages:
    chat_msg = st.chat_message(msg["role"])
    chat_msg.write(msg["content"])

# Capture user input from chat input box
if prompt := st.chat_input("What is up?"):
    # Add the user's message to the conversation
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Fetch the client instance
    client = st.session_state.client

    # Send the chat request to Cohere's chat API (or streaming chat API)
    stream = client.chat_stream(
        model='command-r',
        message=[{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages],  # Flatten the messages into the right format
        temperature=0,       
        max_tokens=1500,
        prompt_truncation='AUTO',
        connectors=[],
        documents=[]
    )

    # Collect the streaming response
    response_text = ""
    if stream:
        for event in stream:
            if event.event_type == "text-generation":
                response_text += event.text  # Collect text from the stream

    # Display the assistant's response
    with st.chat_message("assistant"):
        st.write(response_text)

    # Add the assistant's message to the conversation history
    st.session_state.messages.append({"role": "assistant", "content": response_text})

    # Limit the conversation history to the last 5 messages to keep it manageable
    st.session_state.messages = st.session_state.messages[-5:]
