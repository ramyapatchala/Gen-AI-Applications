import time
import os
import logging
from dotenv import load_dotenv
load_dotenv("test.env")
logging.basicConfig(filename = 'test_models_debug.txt', level=logging.INFO)
question_to_ask = "Why are LLMs (AI) a danger to society?"

system_message = """
Goal: Answer the question using bullets.
      The answer should be appropriate for a 10 year old child to understand
      """

def output_info(content, start_time, model_info):
    end_time = time.time()
    time_taken = end_time - start_time
    time_taken = round(time_taken * 10) / 10

    output = f"For {model_info}, time taken = " + str(time_taken)
    logging.info(output)
    logging.info(f"  --> {content}")
    st.write(output)

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
st.write(content)
output_info(content, start_time, model_info=model)
st.write("done calcs...")
    
