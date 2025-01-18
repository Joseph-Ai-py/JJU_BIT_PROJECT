from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import streamlit as st
from datetime import datetime

def generate_response(query, mmr_docs):
    """
    Generates a response for the given query using the retrieved documents,
    and displays the answer in a streaming fashion (GPT-like).
    """
    print(f"Debug: Generating response for query: {query}")
    print(f"Debug: MMR docs received: {mmr_docs}")

    # Prompt 구성
    question = {"instruction": query, "mmr_docs": mmr_docs}
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                   "Analyze the following and answer the questions in detail. "
                   "Use simple words and answer in a way that a middle school student can understand. "
                   "Answer in Korean."),
        ("human", "query : {instruction}\n\n docs : {mmr_docs}"),
    ])

    # LLM 초기화
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
    chain = prompt | llm

    # 제너레이터 생성
    answer_generator = chain.stream(question)

    # 질문 표시
    st.markdown(f"<p style='font-size:20px;'>{query}</p>", unsafe_allow_html=True)

    # 답변 스트리밍을 위한 placeholder
    placeholder = st.empty()
    partial_text = ""

    # 답변을 토큰 단위로 받아서 실시간 표출
    for token in answer_generator:
        chunk_text = getattr(token, "content", "")
        partial_text += chunk_text
        placeholder.markdown(
            f"<p style='font-size:18px;'>Response: {partial_text}</p>", 
            unsafe_allow_html=True
        )

    print(f"Debug: Final streamed response: {partial_text}")
    return partial_text
