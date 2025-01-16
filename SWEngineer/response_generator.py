from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_teddynote.messages import stream_response
import streamlit as st
import random
from vector_database import query_database

def generate_response(query, mmr_docs):
    """Generates a response for the given query using the retrieved documents."""
    question = {"instruction": query, "mmr_docs": mmr_docs}
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Analyze the following content and answer the question. Answer in Korean."),
        ("human", "{instruction}\n{mmr_docs}"),
    ])

    llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
    chain = prompt | llm
    answer = chain.stream(question)

    st.markdown(f"<p style='font-size:20px;'>{query}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:18px;'>Response: {stream_response(answer, return_output=True)}</p>", unsafe_allow_html=True)

def generate_suggested_questions(global_db, texts):
    """Generates suggested questions based on random text and MMR analysis."""
    suggested_questions = []
    n = random.randint(3, 5)  # Randomly choose number of questions to generate

    for _ in range(n):
        # Randomly select a text
        random_text = random.choice(texts)

        # Perform MMR analysis to retrieve additional relevant texts
        mmr_docs = query_database(global_db, random_text)[:5]

        # Generate a question based on the retrieved documents
        question_prompt = f"""
        [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Analyze the following content and create questions about the content. 
        Answer in Korean.
        Create a question based on the following content: 
        {[doc.page_content for doc in mmr_docs]}
        """
        llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant that creates questions for semantic analysis."),
            ("human", question_prompt),
        ])
        chain = prompt | llm
        suggested_question = chain.run()

        suggested_questions.append(suggested_question)

    return suggested_questions

def display_suggested_questions(suggested_questions):
    """Displays the suggested questions below the text input bar."""
    st.markdown("### Suggested Questions")
    for idx, question in enumerate(suggested_questions, start=1):
        if st.button(f"Question {idx}: {question}"):
            st.session_state["query"] = question
