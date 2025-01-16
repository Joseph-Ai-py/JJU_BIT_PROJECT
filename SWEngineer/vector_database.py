from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

def create_vector_database(texts):
    print("Creating vector database...")
    try:
        embeddings_model = OpenAIEmbeddings()
        print("Initialized OpenAI embeddings model.")

        db = Chroma.from_texts(
            texts,
            embeddings_model,
            collection_name='esg',
            persist_directory='./SWEngineer/db/chromadb',
            collection_metadata={'hnsw:space': 'cosine'},
        )
        print("Vector database created successfully.")
        return db

    except Exception as e:
        print(f"Error during vector database creation: {e}")
        raise

def query_database(db, query):
    print("Querying the vector database...")
    try:
        mmr_docs = db.max_marginal_relevance_search(query, k=5, fetch_k=20)
        print(f"Query returned {len(mmr_docs)} documents.")
        return mmr_docs

    except Exception as e:
        print(f"Error during querying: {e}")
        raise