from dotenv import load_dotenv

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

INDEX_NAME = "langchain-doc-index"


def run_llm(query: str):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    chat = ChatOpenAI(temperature=0)

    template = """
    Answer user questions based solely on the context below.

    <context>
    {context}
    </context>

    If the answer is not provided in the context, say "Answer not in context".

    Question:
    {input}
    """
    prompt = PromptTemplate.from_template(template)

    stuff_documents_chain = create_stuff_documents_chain(
        llm=chat,
        prompt=prompt,
    )

    qa = create_retrieval_chain(
        retriever=docsearch.as_retriever(),
        combine_docs_chain=stuff_documents_chain,
    )

    result = qa.invoke({"input": query})

    return result


if __name__ == "__main__":
    res = run_llm(query="How to make Pizza?")
    print(res.get("answer"))
