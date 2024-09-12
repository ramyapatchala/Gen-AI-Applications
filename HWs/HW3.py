import streamlit as st
from openai import OpenAI
import cohere

st.title("My lab3 Question answering chatbot")

if 'client' not in st.session_state:
    api_key = st.secrets['cohere_key']
    st.session_state.client = cohere.Client(api_key)

if 'messages' not in st.session_state:
    st.session_state['messages'] = [{"role" :"assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    chat_msg = st.chat_message(msg["role"])
    chat_msg.write(msg["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role":"user", "content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    client = st.session_state.client
    stream = client.chat.completions.create(
        model = model_to_use,
        messages = st.session_state.messages,
        stream = True
    )

    with st.chat_message("assistant"):
        response = st.write_stream(stream)
    
    st.session_state.messages.append({"role":"assistant", "content":response})

    st.session_state.messages = st.session_state.messages[-2:]
