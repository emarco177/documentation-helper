import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings


load_dotenv()

# initialize the embeddings and vector store
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    output_dimensionality=1536,
    batch_size=50,
)

# initialize the vector store
vectorstore = PineconeVectorStore(
    index_name=os.getenv("PINECONE_INDEX_NAME"), embedding=embeddings)

# initialize the chat model
model = chat_model = init_chat_model(
    model="gemini-2.5-flash",
    model_provider="google_genai",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.2,
    max_output_tokens=1024,
)

@tool (response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documents to help answer user queries about LangChain."""
    # Retrieve top 4 most similar documents
    retrived_docs = vectorstore.as_retriever().invoke(query, k=4)

    # Seralize documents for the model
    serialized = '\n\n'.join(
        (f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}") 
        for doc in retrived_docs
    )

    # Return the serialized documents and raw documents
    return serialized, retrived_docs


def run_llm(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation.
    
    Args:
        query: The user's question
        
    Returns:
        Dictionary containing:
            - answer: The generated answer
            - context: List of retrieved documents
    """
    # Create the agent with retrieval tool
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )
    
    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)
    
    # Build messages list
    messages = [{"role": "user", "content": query}]
    
    # Invoke the agent
    response = agent.invoke({"messages": messages})
    
    # Extract the answer from the last AI message
    answer = response["messages"][-1].content
    
    # Extract context documents from ToolMessage artifacts
    context_docs = []
    for message in response["messages"]:
        # Check if this is a ToolMessage with artifact
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            # The artifact should contain the list of Document objects
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)
    
    return {
        "answer": answer,
        "context": context_docs
    }

if __name__ == '__main__':
    result = run_llm(query="what are deep agents?")
    print(result)