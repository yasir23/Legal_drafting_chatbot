


import os
import streamlit as st
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
import time
from langchain_community.document_loaders import UnstructuredMarkdownLoader


from typing import Any, Callable, Dict, List, Optional, Union
from langchain.docstore.document import Document
import json

from langchain.document_loaders.base import BaseLoader
from pathlib import Path


from langchain_community.document_loaders import PyPDFLoader

file_path = (
    "Law-on-Drafting-Pleading-Conveyancing-YAL.pdf"
)





if "vector" not in st.session_state:

    st.session_state.embeddings = FastEmbedEmbeddings(model_name = 'BAAI/bge-small-en-v1.5')

    st.session_state.loader = JSONLoader(file_path)
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.documents = st.session_state.text_splitter.split_documents( st.session_state.docs)
    st.session_state.vector = FAISS.from_documents(st.session_state.documents, st.session_state.embeddings)
st.title("Legal drafting")


with st.chat_message("user"):
    st.write("Hello👋 How can Assist you!")



sidebar_logo = "8d33d83eac1cf8e3ea1b4840ccc8baef-removebg-preview.png"






st.logo(sidebar_logo,link=None, icon_image=None)


# "with" notation
with st.sidebar:
    st.title("Prompts")
    st.markdown("Talk to the chatbot!")
    

llm = ChatGroq(
    api_key = st.secrets["GROQ_API_KEY"],
    model_name='mixtral-8x7b-32768'
    )

prompt = ChatPromptTemplate.from_template("""
Act as the mental health bot who will council the user for mental health issues.
According to the context provided.guide him to think in present.
Use the tone of words like a girlfriend and close friend,
but not tell him I am your close friend or girlfriend just use the tone of it in your words.
use the words of love and passion.                  
                                        
I will tip you $200 if the user finds the answer helpful. 
<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)

retriever = st.session_state.vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt = st.chat_input("Input your prompt here")


# If the user hits enter
if prompt:
    # Then pass the prompt to the LLM
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt})
    print(f"Response time: {time.process_time() - start}")



    st.write(response["answer"])

    # # With a streamlit expander
    # with st.expander("Document Similarity Search"):
    #     # Find the relevant chunks
    #     for i, doc in enumerate(response["context"]):
    #         # print(doc)
    #         # # st.write(f"Source Document # {i+1} : {doc.metadata['source'].split('/')[-1]}")
    #         st.write(doc.page_content)
    #         st.write("--------------------------------")
