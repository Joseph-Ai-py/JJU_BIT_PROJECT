
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

def generate_suggested_questions(global_db, texts):
    """Generates suggested questions based on random text and MMR analysis."""
    print("Debug: Starting to generate suggested questions.")
    suggested_questions = []
    n = random.randint(3, 5)  # Randomly choose number of questions to generate

    print(f"Debug: Number of suggested questions to generate: {n}")

    for _ in range(n):
        # Randomly select a text
        random_text = random.choice(texts)
        print(f"Debug: Randomly selected text: {random_text}")

        # Perform MMR analysis to retrieve additional relevant texts
        mmr_docs = query_database(global_db, random_text)[:5]
        print(f"Debug: Retrieved MMR docs: {mmr_docs}")

        # Generate a question based on the retrieved documents
        question_prompt = "Create a question based on the following content: " + "\n".join([doc.page_content for doc in mmr_docs])
        print(f"Debug: Question prompt: {question_prompt}")

        llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o")
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Analyze the following content and create one question related to it. Make sure the question is very easy for users to understand. Answer in Korean."),
            ("human", question_prompt),
        ])
        chain = prompt | llm
        suggested_question = chain.invoke({"question_prompt" : question_prompt})

        print(f"Debug: Generated suggested question: {suggested_question}")

        suggested_questions.append(suggested_question.content)

    return suggested_questions

def display_suggested_questions(suggested_questions):
    """Displays the suggested questions below the text input bar."""
    print(f"Debug: Displaying suggested questions: {suggested_questions}")
    st.sidebar.markdown("### Suggested Questions")

    # 선택된 질문을 session_state에 저장 (없으면 None으로 초기화)
    if "selected_question" not in st.session_state:
        st.session_state.selected_question = None

    # selectbox에서 이전에 선택된 질문을 유지하도록 설정
    selected_question = st.sidebar.selectbox(
        "Select a question", 
        suggested_questions, 
        index=suggested_questions.index(st.session_state.selected_question) if st.session_state.selected_question in suggested_questions else 0,
        key="query_select",
        
    )

    # 선택된 질문을 session_state에 계속 유지
    if selected_question:
        st.session_state.selected_question = selected_question
        st.session_state.query = selected_question
        print(f"query : {selected_question}")
        
    return st.session_state.selected_question