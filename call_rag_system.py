


import os 

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory


from scraper import scrape_all_books

def rag_system(base_url, max_pages=4, google_api_key="YOUR_API_KEY"):
    persist_directory = "chroma_db2"
    if os.path.exists(persist_directory):
        print("Loading existing Chroma database...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        print("Creating new Chroma database...")
        all_book_data = scrape_all_books(base_url, max_pages)

        documents = []
        for book in all_book_data:
            documents.append(
                f"Title: {book['title']}\nDescription: {book['full_description']}\nPrice: {book['price']}\nCategory: {book['category']}\nAvailability: {book['availability']}"
            )

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.create_documents(documents)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
        os.makedirs(persist_directory, exist_ok=True)
        db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)
    retriever = db.as_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(llm, retriever, memory=memory)

    return qa
