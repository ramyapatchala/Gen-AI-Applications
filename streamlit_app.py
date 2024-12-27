import streamlit as st
from openai import OpenAI
import os

page1 = st.Page("Document Q&A System.py", title="Document Q&A System")
page2 = st.Page("Multi Webpage Summarizer.py", title="Multi Webpage Summarizer")
page3 = st.Page("Chatbot.py", title="Chatbot")
page4 = st.Page("Retreived Augmented Generation (RAG).py", title="Retreived Augmented Generation (RAG)")
page5 = st.Page("iSchool Club Agent.py", title="iSchool Club Agent")
page6 = st.Page("News Bot.py", title = "News Bot")

pg = st.navigation([page1, page2, page3, page4, page5, page6])
st.set_page_config(page_title="Gen AI Applications")
pg.run()
