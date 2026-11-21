import asyncio
import os
import re
import ssl
from typing import Any, Dict, List

import certifi
import requests
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import (Colors, log_error, log_header, log_info, log_success,
                    log_warning)

load_dotenv()

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=False,
    chunk_size=50,
    retry_min_seconds=10,
)
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
# vectorstore = PineconeVectorStore(
#     index_name="langchain-docs-2025", embedding=embeddings
# )
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()

LLMS_TXT = "https://docs.langchain.com/llms.txt"


def get_documentation_urls() -> List[str]:
    """Find every LangChain Python docs page through the site's llms.txt."""
    llms_txt = requests.get(LLMS_TXT, timeout=30).text
    python_line = next(line for line in llms_txt.splitlines() if "Build / Python" in line)
    python_index = re.search(r"\((https://\S+?\.txt)\)", python_line).group(1)
    index = requests.get(python_index, timeout=30).text
    urls = re.findall(r"\((https://docs\.langchain\.com/\S+?\.md)\)", index)
    return [url for url in urls if "/oss/python/" in url]


def chunk_urls(urls: List[str], chunk_size: int = 20) -> List[List[str]]:
    """Split the URLs into batches for TavilyExtract."""
    return [urls[i : i + chunk_size] for i in range(0, len(urls), chunk_size)]


async def extract_batch(urls: List[str], batch_num: int) -> List[Dict[str, Any]]:
    """Extract one batch of URLs with TavilyExtract."""
    log_info(f"🔄 TavilyExtract: Processing batch {batch_num} with {len(urls)} URLs")
    response = await tavily_extract.ainvoke(input={"urls": urls})
    results = response.get("results", [])
    log_success(f"TavilyExtract: Completed batch {batch_num} - {len(results)} pages")
    return results


async def async_extract(url_batches: List[List[str]]) -> List[Document]:
    """Extract all batches concurrently and turn each page into a LangChain Document."""
    log_header("DOCUMENT EXTRACTION PHASE")
    tasks = [extract_batch(batch, i + 1) for i, batch in enumerate(url_batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_docs = []
    for batch_num, batch_results in enumerate(results, start=1):
        if isinstance(batch_results, Exception):
            log_error(f"TavilyExtract: Batch {batch_num} failed - {batch_results}")
            continue
        for result in batch_results:
            if not result.get("raw_content"):
                log_warning(f"TavilyExtract: Skipping {result.get('url')} - no content")
                continue
            all_docs.append(
                Document(
                    page_content=result["raw_content"],
                    metadata={"source": result["url"]},
                )
            )
    log_success(f"TavilyExtract: Extracted {len(all_docs)} documents")
    return all_docs


async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """Process documents in batches asynchronously."""
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"📚 VectorStore Indexing: Preparing to add {len(documents)} documents to vector store",
        Colors.DARKCYAN,
    )

    # Create batches
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"📦 VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )

    # Process all batches concurrently
    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await asyncio.to_thread(vectorstore.add_documents, batch)
            log_success(
                f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
            )
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True

    # Process batches concurrently
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successful batches
    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully"
        )


async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(f"📄 llms.txt: Reading the documentation index from {LLMS_TXT}", Colors.PURPLE)
    urls = get_documentation_urls()
    log_success(f"llms.txt: Found {len(urls)} documentation pages")

    url_batches = chunk_urls(urls, chunk_size=20)
    all_docs = await async_extract(url_batches)

    # Split documents into chunks
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"✂️  Text Splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap",
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )

    # Process documents asynchronously
    await index_documents_async(splitted_docs, batch_size=500)

    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • URLs found: {len(urls)}")
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(splitted_docs)}")


if __name__ == "__main__":
    asyncio.run(main())
