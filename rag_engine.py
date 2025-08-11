import json
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq.chat_models import ChatGroq
from langchain.docstore.document import Document
from stt import transcribe
from dotenv import load_dotenv

load_dotenv()

# API key from environment
GROQ_API = os.getenv("GROQ_API_KEY")

# Global models - initialize once
embedding_model = None
llm = None
vectordb = None

def initialize_models():
    """Initialize models once"""
    global embedding_model, llm, vectordb
    
    if embedding_model is None:
        print("Initializing embedding model...")
        # Initialize embedding model
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("✓ Embedding model loaded")
    
    if llm is None:
        print("Initializing LLM...")
        # Initialize LLM
        if not GROQ_API:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=GROQ_API)
        print("✓ LLM loaded")
    
    if vectordb is None:
        print("Loading knowledge base...")
        # Load and process knowledge base
        if os.path.exists("tirumala_english_cleaned.json"):
            with open("tirumala_english_cleaned.json", "r", encoding="utf-8") as f:
                json_data = json.load(f)
            
            print(f"Loaded {len(json_data)} documents from knowledge base")
            
            # Convert to LangChain Document format - handle your specific JSON structure
            documents = []
            for item in json_data:
                if isinstance(item, dict):
                    # Use cleaned_content field as the main content
                    if 'cleaned_content' in item:
                        content = str(item['cleaned_content'])
                        # Optionally include URL as metadata
                        metadata = {'url': item.get('url', '')} if 'url' in item else {}
                        documents.append(Document(page_content=content, metadata=metadata))
                    elif 'content' in item:
                        # Fallback to 'content' field
                        content = str(item['content'])
                        documents.append(Document(page_content=content))
                    else:
                        # If neither field exists, combine all fields
                        content = " ".join([f"{k}: {v}" for k, v in item.items() if isinstance(v, (str, int, float))])
                        if content.strip():
                            documents.append(Document(page_content=content))
                elif isinstance(item, str):
                    # If item is just a string
                    documents.append(Document(page_content=item))
                else:
                    # If item is something else, convert to string
                    content = str(item)
                    if content.strip():
                        documents.append(Document(page_content=content))
            
            print(f"Processed {len(documents)} valid documents")
            
            if not documents:
                raise ValueError("No valid documents found in the JSON file. Please check the file structure.")
            
            # Split large documents if needed
            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = splitter.split_documents(documents)
            
            print(f"Split into {len(docs)} chunks")
            
            # Build FAISS vector store
            vectordb = FAISS.from_documents(docs, embedding_model)
            print("✓ Vector database created")
        else:
            raise FileNotFoundError("tirumala_english_cleaned.json not found. Please ensure the knowledge base file exists.")

def get_rag_response(question: str) -> str:
    """Get RAG response for a text question"""
    initialize_models()
    
    # Create prompt template
    template = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are an assistant for AR App. Use the context provided to answer in brief the question based on given Knowledge bank.
        Context:
        {context}
        Question: {question}
        Answer:"""
    )
    
    # Build RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(),
        chain_type_kwargs={"prompt": template, "document_variable_name": "context"}
    )
    
    # Run QA chain
    answer = qa_chain.run(question)
    return answer

def build_rag_pipeline(audio_bytes: bytes) -> tuple[str, str]:
    """Complete RAG pipeline: audio -> transcription -> answer"""
    # 1. Transcribe audio input to question text
    question = transcribe(audio_bytes)
    
    # 2. Get RAG response
    answer = get_rag_response(question)
    
    return question, answer