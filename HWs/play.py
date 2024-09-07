import cohere
import streamlit as st
cohere_key = st.secrets['cohere_key']

def do_cohere():
    cohere_key = st.secrets['cohere_key']
    co = cohere.Client(cohere_key)
    response = co.tokenize(
        text = system_message, model='command')
    events = co.chat_stream(
                    model='command-r',
                    message=question_to_ask,
                    temperature=0,       
                    max_tokens=1500,
                    chat_history=[{"role":"SYSTEM", "message":system_message}],
                    prompt_truncation='AUTO',
                    connectors=[],
                    documents=[]
    )

    response_text=""
    for event in events:
        if event.event_type=="text-generation":
            response_text = response_text + str(event.text)
    return response_text

model="Cohere"
start_time = time.time()
content = do_cohere()
output_info(content, start_time, model_info=model)
st.write("done calcs...")
    
