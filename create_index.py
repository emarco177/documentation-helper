import os

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

if __name__ == "__main__":
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = "langchain-docs"

    if pc.has_index(index_name):
        print(f"Index '{index_name}' already exists")
    else:
        pc.create_index(
            name=index_name,
            dimension=1536,  # the size of OpenAI's text-embedding-3-small vectors
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"Index '{index_name}' created")
