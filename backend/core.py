from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from typing import Any, Dict, List
from langchain.chains import ConversationalRetrievalChain
from langchain_pinecone import PineconeVectorStore


INDEX_NAME = "langchain-doc-index"


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docsearch = PineconeVectorStore(embedding=embeddings, index_name=INDEX_NAME)

    chat = ChatOpenAI(
        verbose=True,
        temperature=0,
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=docsearch.as_retriever(), return_source_documents=True
    )
    return qa.invoke({"question": query, "chat_history": chat_history})
