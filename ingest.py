"""
Automatic Media Generator - Ingestion Module (1.0) 
Created and Developed by: Kyle Zheng   


Basic Pathway: 
1. The system ingests a variety of dental-related content from multiple sources, including PDFs, websites, and text files.
2. The ingested content is processed and stored in a vector database, making it easily retrievable for the content generation phase.
3. This module ensures that the content is organized and accessible, providing a rich knowledge base for the ChatOllama model to generate accurate and engaging social media posts. 

"""


# Import Modules and Libraries 
import os 
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma 
from langchain_community.embeddings import HuggingFaceEmbeddings   
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv 


# Load environment variables
load_dotenv()


# [DEBUG/TEST] Environment Verification Test
if os.getenv("LANGSMITH_API_KEY"):
    print("✅ LangSmith API Key Loaded")
else:
    print("❌ LangSmith API Key Missing")


def intialize_rag():
    """ Initializes RAG (Retrieval-Augmented Generation) by loading and processing the content from the specified directory, and storing it in a vector database for efficient retrieval during content generation."""
    
    # (1). Load the data, use different loaders for specfiic file types (PDFs, text files, etc.) and specify the directory containing the content.    
    txt_loader = DirectoryLoader(  # TXT file loader 
        "./knowledge_base", 
        glob="**/*.txt",
        loader_cls=TextLoader  
    )

    pdf_loader = DirectoryLoader(  # PDF file loader 
        "./knowledge_base", 
        glob="**/*.pdf",
        loader_cls=PyPDFLoader  
    )

    documents = txt_loader.load() + pdf_loader.load()  # Combine loaded documents from both loaders into a single list
    
    # 1.5 [DEBUG] Check if documents were loaded successfully   
    if documents is not None:
        print(f"[DEBUG | SUCCESS] Loaded {len(documents)} documents from the knowledge base. ")
    else: 
        print("[DEBUG | ERROR] No documents found in the knowledge base. Please check the directory and file formats. ")
        return None


    # (2). Split the loaded documents into smaller chunks for better processing and retrieval.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, 
        add_start_index=True)  # Add start index to keep track of original document positions    
    split_docs = text_splitter.split_documents(documents)

    # 2.5 [DEBUG] Check if documents were split successfully
    if split_docs is not None:
        print(f"[DEBUG | SUCCESS] Split documents into {len(split_docs)} chunks for better processing. ")
    else: 
        print("[DEBUG | ERROR] Document splitting failed. Please check the text splitter configuration. ")
        return None
    

    # (3). Create a local vector database using Chroma to store the processed document chunks, and use HuggingFaceEmbeddings to generate embeddings for the chunks. 
    embeddings = HuggingFaceEmbeddings(model_name= "all-MiniLM-L6-v2")  # Use a smaller model for faster embedding generation
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory="./chroma_db" # Saves to a local directory
    )   
    print("[DEBUG | SUCCESS] Vector database created and persisted locally at './chroma_db'. ")


if __name__ == "__main__":
    intialize_rag() 
