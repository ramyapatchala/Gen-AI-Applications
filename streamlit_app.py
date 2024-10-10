import streamlit as st
from openai import OpenAI
import os

page1 = st.Page("HWs/HW1.py", title="Document Q&A System")
page2 = st.Page("HWs/HW2.py", title="Multi Webpage Summarizer")
page3 = st.Page("HWs/HW3.py", title="Chatbot")
page4 = st.Page("HWs/HW4.py", title="Retreived Augmented Generation (RAG)")
page5 = st.Page("HWs/HW5.py", title="iSchool Club Chatbot")
page6 = st.Page("HWs/HW6.py", title = "HW 6")

pg = st.navigation([page1, page2, page3, page4, page5, page6])
st.set_page_config(page_title="HW Manager")
pg.run()
