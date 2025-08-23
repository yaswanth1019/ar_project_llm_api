import json
import os
import pickle
from pathlib import Path
from typing import List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq.chat_models import ChatGroq
from langchain.docstore.document import Document
from dotenv import load_dotenv

load_dotenv()

# Configuration
GROQ_API = os.getenv("GROQ_API_KEY")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_STORE_PATH = "vectorstore.pkl"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Global models - lazy initialization
embedding_model = None
llm = None
vectordb = None

def get_embedding_model():
    """Lazy initialization of embedding model"""
    global embedding_model
    if embedding_model is None:
        print("Loading embedding model...")
        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'},  # Force CPU to save memory
            encode_kwargs={'normalize_embeddings': True}
        )
        print("✓ Embedding model loaded")
    return embedding_model

def get_llm():
    """Lazy initialization of LLM"""
    global llm
    if llm is None:
        print("Initializing LLM...")
        if not GROQ_API:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        llm = ChatGroq(
            model_name="llama-3.1-8b-instant", 
            groq_api_key=GROQ_API,
            temperature=0.1,
            max_tokens=200  # Limit response length
        )
        print("✓ LLM loaded")
    return llm

def load_or_create_vectorstore():
    """Load existing vectorstore or create new one"""
    global vectordb
    
    if vectordb is not None:
        return vectordb
    
    # Try to load existing vectorstore
    if os.path.exists(VECTOR_STORE_PATH):
        try:
            print("Loading existing vectorstore...")
            with open(VECTOR_STORE_PATH, 'rb') as f:
                vectordb = pickle.load(f)
            print("✓ Vectorstore loaded from cache")
            return vectordb
        except Exception as e:
            print(f"Failed to load cached vectorstore: {e}")
            print("Creating new vectorstore...")
    
    # Create new vectorstore
    embedding_model = get_embedding_model()
    
    # Load knowledge base
    knowledge_file = "tirumala_english_cleaned.json"
    if not os.path.exists(knowledge_file):
        raise FileNotFoundError(f"{knowledge_file} not found")
    
    with open(knowledge_file, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    
    print(f"Processing {len(json_data)} documents...")
    
    # Convert to documents more efficiently
    documents = []
    for i, item in enumerate(json_data):
        try:
            content = ""
            metadata = {"source": f"doc_{i}"}
            
            if isinstance(item, dict):
                # Extract content
                if 'cleaned_content' in item:
                    content = str(item['cleaned_content']).strip()
                elif 'content' in item:
                    content = str(item['content']).strip()
                else:
                    # Combine all text fields
                    content = " ".join([
                        str(v) for k, v in item.items() 
                        if isinstance(v, (str, int, float)) and len(str(v).strip()) > 10
                    ])
                
                # Add URL to metadata if available
                if 'url' in item:
                    metadata['url'] = item['url']
            
            elif isinstance(item, str):
                content = item.strip()
            
            # Only add if content is substantial
            if content and len(content) > 50:
                documents.append(Document(page_content=content, metadata=metadata))
                
        except Exception as e:
            print(f"Error processing document {i}: {e}")
            continue
    
    if not documents:
        raise ValueError("No valid documents found")
    
    print(f"Created {len(documents)} documents")
    
    # Split documents efficiently
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    splits = splitter.split_documents(documents)
    print(f"Split into {len(splits)} chunks")
    
    # Create vectorstore
    print("Creating vectorstore...")
    vectordb = FAISS.from_documents(splits, embedding_model)
    
    # Save vectorstore for future use
    try:
        with open(VECTOR_STORE_PATH, 'wb') as f:
            pickle.dump(vectordb, f)
        print("✓ Vectorstore saved to cache")
    except Exception as e:
        print(f"Warning: Could not cache vectorstore: {e}")
    
    print("✓ Vectorstore created")
    return vectordb

def initialize_models():
    """Initialize all models"""
    print("Initializing RAG models...")
    get_embedding_model()
    get_llm()
    load_or_create_vectorstore()
    print("✓ All RAG models initialized")

def get_rag_response(question: str, max_context_length: int = 2000) -> str:
    """Get RAG response with optimized context length"""
    try:
        # Ensure models are initialized
        llm = get_llm()
        vectordb = load_or_create_vectorstore()
        
        # Create optimized prompt
        template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Based on the provided context, answer the question briefly and accurately.
            
Context: {context}

Question: {question}

Answer (be concise and helpful):"""
        )
        
        # Create retriever with limited results
        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}  # Limit to top 3 results
        )
        
        # Build QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={
                "prompt": template, 
                "document_variable_name": "context"
            },
            return_source_documents=False  # Don't return sources to save memory
        )
        
        # Get response
        result = qa_chain.run(question)
        
        # Clean up response
        if isinstance(result, str):
            return result.strip()
        else:
            return str(result).strip()
        
    except Exception as e:
        print(f"RAG error: {e}")
        return f"I apologize, but I encountered an error while processing your question: {str(e)}"

def build_rag_pipeline(audio_bytes: bytes) -> tuple[str, str]:
    """Complete RAG pipeline: audio -> transcription -> answer"""
    # Import here to avoid circular imports
    from stt import transcribe
    
    # 1. Transcribe audio input to question text
    question = transcribe(audio_bytes)
    
    # 2. Get RAG response
    answer = get_rag_response(question)
    
    return question, answer

# Utility function to clear cache
def clear_vectorstore_cache():
    """Clear the vectorstore cache"""
    if os.path.exists(VECTOR_STORE_PATH):
        os.remove(VECTOR_STORE_PATH)
        print("Vectorstore cache cleared")
        
    global vectordb
    vectordb = None