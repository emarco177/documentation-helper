from dotenv import load_dotenv
import os

from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader

from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Two Karpathy lecture videos
#urls = ["https://youtu.be/kCc8FmEb1nY", "https://youtu.be/VMj-3S1tku0"]
urls = ["https://youtu.be/kCc8FmEb1nY"]


# Directory to save audio files
save_dir = "/Users/maximeberthelot/Downloads/YouTube"

# Check if transcript file exists
if os.path.exists(os.path.join(save_dir, "transcript.txt")):
    with open(os.path.join(save_dir, "transcript.txt"), "r") as f:
        text = f.read()
else:
    # Transcribe the videos to text
    loader = GenericLoader(YoutubeAudioLoader(urls, save_dir), OpenAIWhisperParser(api_key=OPENAI_API_KEY))
    docs = loader.load()
    #print(docs)  # Add this line to check if the docs list is empty

    # Combine doc
    combined_docs = [doc.page_content for doc in docs]
    text = " ".join(combined_docs)

    # Save transcript to file
    with open(os.path.join(save_dir, "transcript.txt"), "w") as f:
        f.write(text)

# Split them
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
splits = text_splitter.split_text(text)

# Build an index
embeddings = OpenAIEmbeddings()
vectordb = FAISS.from_texts(splits, embeddings)

# Build a QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
    chain_type="stuff",
    retriever=vectordb.as_retriever(),
)

# Ask a question!
query = "Why do we need to zero out the gradient before backprop at each step?"
answer = qa_chain.run(query)
print(answer)