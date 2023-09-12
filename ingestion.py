import os
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

pinecone.init(api_key=os.environ['PINECONE_API_KEY'], environment=os.environ['PINECONE_ENVIRONMENT_REGION'])

def ingest_docs():
    loader = ReadTheDocsLoader(
        path='langchain-docs/api.python.langchain.com/en/latest')
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", "", "\t"])
    documents = text_splitter.split_documents(documents=raw_documents)

    for document in documents:
        old_path = document.metadata['source']
        new_url = old_path.replace('langchain-docs', 'https:/')
        document.metadata.update({'source': new_url})
        
    print(f"Going to insert {len(documents)} Pinecone")
    enbeddings = OpenAIEmbeddings()
    Pinecone.from_documents(documents=documents, embedding=enbeddings, index_name='langchain-doc-index')
    print('persisted to pinecone')

if __name__ == '__main__':
    ingest_docs()
