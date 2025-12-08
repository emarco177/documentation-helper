import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyMap, TavilyExtract


from logger import (Colors, log_error, log_header, log_info, log_success,
                    log_warning)
load_dotenv()

# Configure SSL context to use certifi certificates
ssel_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT__FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)
#chroma = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
vectorstore = PineconeVectorStore(index_name="langchain-docs-2025", embedding=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()




async def main():
    """Main async function to orchestrate the entire process."""

    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "TavilyCrawl: Starting to Crawl documentation from https://python.langchain.com/",
        Colors.PURPLE
    )

    # Create batches
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)   
    ]

    # Process all batches concurrently
    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vetorstore.aadd_documents(batch)
            log_success(
                f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
            )

        except Exception as e:
            log_error(
                f"VectorStore Indexing: Failed to add batch {batch_num} - {e}"
            )
            return False

        return True








if __name__ == "__main__":
    asyncio.run(main())
