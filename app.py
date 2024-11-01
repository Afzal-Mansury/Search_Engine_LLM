import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

# create the Wrapper | update
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

#Open Search
search = DuckDuckGoSearchRun(name = "Search")


## App TITLE
st.title("Langchain - Chat with searcg")

#Sidebar for Settings
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your Groq API keys:",type="password")

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assitant","content":"Hi, I am a chatbot who can search from Arxiv, Wiki and Google"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Create Prompt 
if prompt:=st.chat_input(placeholder="What is Machine Learning?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key=api_key, model_name = "llama3-8b-8192",streaming=True)
    tools = [search, arxiv, wiki]

    search_agent = initialize_agent(tools,llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_errors = True)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks= [st_cb])
        st.session_state.messages.append({"role":"assitant","content":response})
        st.write(response)
