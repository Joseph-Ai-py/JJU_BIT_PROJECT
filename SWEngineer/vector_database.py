import os
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

def create_vector_database(texts):
    print("Creating vector database...")

    # 디렉토리 확인 및 생성
    persist_directory = './SWEngineer/db/chromadb'
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
        print(f"Created directory: {persist_directory}")

    try:
        # OpenAI Embeddings 초기화
        embeddings_model = OpenAIEmbeddings()
        print("Initialized OpenAI embeddings model.")

        # Chroma 데이터베이스 생성
        db = Chroma.from_texts(
            texts=texts,
            embedding=embeddings_model,
            collection_name='esg',
            persist_directory=persist_directory,
            collection_metadata={'hnsw:space': 'cosine'},
        )
        print("Vector database created successfully.")
        return db

    except Exception as e:
        print(f"Error during vector database creation: {e}")
        return None

    except Exception as e:
        print(f"Error during vector database creation: {e}")
        raise

def query_database(db, query):
    print("Querying the vector database...")
    try:
        mmr_docs = db.max_marginal_relevance_search(query, k=20, fetch_k=100)
        print(f"Query returned {len(mmr_docs)} documents. \n docs : {mmr_docs}")
        return [docs.page_content for docs in mmr_docs]

    except Exception as e:
        print(f"Error during querying: {e}")
        raise