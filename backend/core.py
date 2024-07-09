from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import PromptTemplate

load_dotenv()

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore


from langchain_openai import ChatOpenAI, OpenAIEmbeddings


INDEX_NAME = "langchain-doc-index"


def run_llm(query: str):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    chat = ChatOpenAI(verbose=True, temperature=0)
    chat = Ollama(model="llama3")

    retrieval_qa_chat_prompt: PromptTemplate = hub.pull(
        "langchain-ai/retrieval-qa-chat",
    )

    template = """
    Answer any use questions based solely on the context below:
    
    <context>
    {context}
    </context>
    
    if the answer is not provided in the context say "Answer not in context"
    Question:
    {input}
    """
    retrieval_qa_chat_prompt2 = PromptTemplate.from_template(template=template)
    stuff_documents_chain = create_stuff_documents_chain(
        chat, retrieval_qa_chat_prompt2
    )

    qa = create_retrieval_chain(
        retriever=docsearch.as_retriever(), combine_docs_chain=stuff_documents_chain
    )
    result = qa.invoke(input={"input": query})
    return result


if __name__ == "__main__":
    # res = run_llm(query="What is a LangChain Chain?")
    res = run_llm(query="How to make Pizza?")

    print(res["answer"])
