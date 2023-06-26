import os 
from langchain.document_loaders import ReadTheDocsLoader



def ingest_docs() -> None:
   loader = ReadTheDocsLoader(path="langchain-docs/langchain.readthedocs.io/en/latest")
   raw_documents = loader.load()


if __name__ ==  '__main__':
    ingest_docs()


