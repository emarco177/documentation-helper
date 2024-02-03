from dotenv import load_dotenv

load_dotenv()

import os

from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone as PineconeLangChain
from pinecone import Pinecone


pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY"),
)

INDEX_NAME = "langchain-doc-index"


def ingest_docs():
    loader = ReadTheDocsLoader(
        "langchain-docs/api.python.langchain.com/en/latest/chains"
    )

    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print(f"Going to add {len(documents)} to Pinecone")
    PineconeLangChain.from_documents(documents, embeddings, index_name=INDEX_NAME)
    print("****Loading to vectorstore done ***")


if __name__ == "__main__":
    ingest_docs()
