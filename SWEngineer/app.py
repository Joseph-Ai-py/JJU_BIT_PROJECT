import os
import random
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st
from process_pdf import process_pdf
from vector_database import create_vector_database, query_database
from response_generator import generate_response
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

def main_streamlit():
    load_dotenv()

    st.set_page_config(page_title="Semantic Analysis", page_icon="🔬", layout="wide")

    st.sidebar.title("Options")
    st.sidebar.write("Upload a file and enter your query to analyze it.")

    uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

    st.markdown("""
    <h3 style='text-align:right;'>Jeonju-University</h2>
    """, unsafe_allow_html=True)

    st.title("🔬 Semantic Analysis with LangChain")

    st.markdown("""
    This application allows you to upload a PDF, analyze its content, and retrieve information using natural language queries.

    **Features:**
    - Semantic chunking for better document understanding.
    - Vector database for fast and accurate search.
    - Responses in Korean with typing animation.
    """)

    if 'global_db' not in st.session_state or "texts" not in st.session_state:
        st.session_state.global_db = None
        st.session_state.texts = None

    if uploaded_file:
        temp_file_path = f"temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())
        print(f"Uploaded file saved as {temp_file_path}.")

        if st.session_state.texts is None:
            st.session_state.texts = process_pdf(temp_file_path)

        if st.session_state.global_db is None:
            try:
                st.session_state.global_db = create_vector_database(st.session_state.texts)
            except Exception as e:
                print(f"An error occurred during file processing: {e}")

    # 중앙 아래에 입력 및 버튼 배치
    placeholder = st.empty()  # 중앙 하단 위치에 요소를 추가할 공간
    with placeholder.container():
        st.markdown("<div style='height: 10vh;'></div>", unsafe_allow_html=True)  # 위 공간 확보
        cols = st.columns([5, 1])  # 5:1 비율로 컬럼 생성
        with cols[0]:
            query = st.text_input("Enter your query:", key="query_input", value=st.session_state.query)
            print(f"input query : {query}")
        with cols[1]:
            query_button = st.button("Send Query")

    if query_button:
        if st.session_state.global_db is None:
            st.warning("Please upload a file to create the database first.")
        if query is "":
            st.warning("Query cannot be empty.")
        else:
            try:
                print(f"query : {query}")
                mmr_docs = query_database(st.session_state.global_db, query)
                generate_response(query, mmr_docs)
            except Exception as e:
                st.error(f"An error occurred during querying: {e}")

if __name__ == "__main__":

    import pysqlite3
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    import sqlite3

    main_streamlit()