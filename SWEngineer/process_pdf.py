from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
import os

def process_pdf(pdf_filepath):
    print("Processing PDF file...")
    try:
        loader = PyMuPDFLoader(pdf_filepath)
        pages = loader.load()
        print(f"Loaded {len(pages)} pages from the PDF.")

        text_splitter = SemanticChunker(
            OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"), model="text-embedding-3-large"),
            breakpoint_threshold_type="standard_deviation",
            breakpoint_threshold_amount=1.25,
        )
        print("Initialized semantic chunker.")

        texts = []
        for page_number, page in enumerate(pages, start=1):
            print(page.page_content)
            chunks = text_splitter.split_text(page.page_content)
            texts.extend(chunks)
            print(f"Processed page {page_number} with {len(chunks)} chunks.")

        print("PDF processing complete.")
        return texts

    except Exception as e:
        print(f"Error during PDF processing: {e}")
        raise
