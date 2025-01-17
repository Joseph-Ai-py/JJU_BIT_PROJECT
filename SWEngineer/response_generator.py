
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_teddynote.messages import stream_response
import streamlit as st
import random
from datetime import datetime
from vector_database import query_database

def generate_response(query, mmr_docs):
    """Generates a response for the given query using the retrieved documents."""
    print(f"Debug: Generating response for query: {query}")
    print(f"Debug: MMR docs received: {mmr_docs}")

    question = {"instruction": query, "mmr_docs": mmr_docs}
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Analyze the following content and answer the question. Answer in Korean."),
        ("human", "{instruction}\n{mmr_docs}"),
    ])

    llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
    chain = prompt | llm
    answer = chain.stream(question)

    print(f"Debug: Generated response: {answer}")

    st.markdown(f"<p style='font-size:20px;'>{query}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:18px;'>Response: {stream_response(answer, return_output=True)}</p>", unsafe_allow_html=True)