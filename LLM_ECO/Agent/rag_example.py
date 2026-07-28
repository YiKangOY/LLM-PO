"""
RAG System with LangChain Components and LangGraph
Following the official LangChain RAG tutorial patterns
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path

# LangChain core imports
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Document loading and processing
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# OpenAI integrations
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# LangGraph imports
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict

# Set OPENAI_API_KEY and OPENAI_BASE_URL in your environment before running.

class RAGState(TypedDict):
    """State for the RAG workflow"""
    question: str
    context: List[Document]
    answer: str

class LangChainRAGSystem:
    """RAG System using LangChain components with LangGraph orchestration"""
    
    def __init__(self, persist_dir: str = "./vector_store"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(exist_ok=True)
        
        # Initialize LangChain components
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        openai_api_base = os.environ.get("OPENAI_BASE_URL")
        embeddings_kwargs = {
            "model": "text-embedding-ada-002",
        }
        if openai_api_key:
            embeddings_kwargs["openai_api_key"] = openai_api_key
        if openai_api_base:
            embeddings_kwargs["openai_api_base"] = openai_api_base
        self.embeddings = OpenAIEmbeddings(**embeddings_kwargs)
        
        llm_kwargs = {
            "model": "gpt-3.5-turbo-0125",
            "temperature": 0.1,
        }
        if openai_api_key:
            llm_kwargs["openai_api_key"] = openai_api_key
        if openai_api_base:
            llm_kwargs["openai_api_base"] = openai_api_base
        self.llm = ChatOpenAI(**llm_kwargs)
        
        # Initialize vector store
        self.vector_store = InMemoryVectorStore(self.embeddings)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,  # Track position in original document
            separators=["\n\n", "\n", " ", ""]  # Split hierarchy
        )
        
        # Initialize prompt template
        self.prompt_template = PromptTemplate.from_template(
            """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Context:
{context}

Question: {question}

Answer:"""
        )
        
        # Create LangGraph workflow
        self.workflow = self._create_workflow()
    
    def load_and_process_pdf(self, pdf_path: str, pages: Optional[List[int]] = None) -> List[Document]:
        """
        Load PDF and process into documents
        
        Args:
            pdf_path: Path to PDF file
            pages: Specific pages to load (0-indexed). If None, loads all pages
        """
        print(f"Loading PDF: {pdf_path}")
        
        # Use LangChain's PyPDFLoader
        loader = PyPDFLoader(pdf_path)
        
        if pages is not None:
            # Load specific pages
            documents = []
            all_docs = loader.load()
            for page_num in pages:
                if 0 <= page_num < len(all_docs):
                    doc = all_docs[page_num]
                    doc.metadata["page"] = page_num
                    documents.append(doc)
        else:
            # Load all pages
            documents = loader.load()
            # Add page numbers to metadata
            for i, doc in enumerate(documents):
                doc.metadata["page"] = i
        
        print(f"Loaded {len(documents)} pages")
        return documents
    
    def create_database_from_pdf(self, pdf_path: str, pages: Optional[List[int]] = None):
        """Create vector database from PDF"""
        # Load documents
        documents = self.load_and_process_pdf(pdf_path, pages)
        
        if not documents:
            print("No documents loaded")
            return False
        
        # Split documents into chunks
        print("Splitting documents into chunks...")
        all_splits = self.text_splitter.split_documents(documents)
        print(f"Created {len(all_splits)} chunks")
        
        # Add documents to vector store
        print("Creating embeddings and storing in vector database...")
        document_ids = self.vector_store.add_documents(all_splits)
        print(f"Stored {len(document_ids)} document chunks")
        
        return True
    
    def _create_workflow(self) -> StateGraph:
        """Create LangGraph workflow for RAG"""
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("retrieve", self.retrieve_documents)
        workflow.add_node("generate", self.generate_response)
        
        # Add edges manually
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate")
        
        return workflow.compile()
    
    def retrieve_documents(self, state: RAGState) -> RAGState:
        """Retrieve relevant documents using vector similarity search"""
        question = state["question"]
        print(f"Retrieving documents for: {question}")
        
        # Perform similarity search
        retrieved_docs = self.vector_store.similarity_search(
            question,
            k=4  # Return top 4 most similar documents
        )
        
        print(f"Retrieved {len(retrieved_docs)} documents")
        
        return {"context": retrieved_docs}
    
    def generate_response(self, state: RAGState) -> RAGState:
        """Generate response using retrieved context"""
        question = state["question"]
        context_docs = state["context"]
        
        print("Generating response...")
        
        # Prepare context from retrieved documents
        context = "\n\n".join([
            f"Page {doc.metadata.get('page', 'Unknown')}:\n{doc.page_content}" 
            for doc in context_docs
        ])
        
        # Create the prompt
        messages = self.prompt_template.invoke({
            "question": question,
            "context": context
        })
        
        # Generate response
        response = self.llm.invoke(messages)
        
        return {"answer": response.content}
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system"""
        initial_state = RAGState(
            question=question,
            context=[],
            answer=""
        )
        
        final_state = self.workflow.invoke(initial_state)
        return final_state
    
    def save_vector_store(self):
        """Save vector store state (for InMemoryVectorStore, we'll pickle it)"""
        import pickle
        
        store_data = {
            "docstore": self.vector_store.store,
            "index_to_docstore_id": getattr(self.vector_store, 'index_to_docstore_id', {})
        }
        
        with open(self.persist_dir / "vector_store.pkl", "wb") as f:
            pickle.dump(store_data, f)
        
        print(f"Vector store saved to {self.persist_dir}")
    
    def load_vector_store(self) -> bool:
        """Load vector store state"""
        import pickle
        
        store_path = self.persist_dir / "vector_store.pkl"
        if not store_path.exists():
            return False
        
        try:
            with open(store_path, "rb") as f:
                store_data = pickle.load(f)
            
            # Recreate vector store with loaded data
            self.vector_store = InMemoryVectorStore(self.embeddings)
            self.vector_store.store = store_data["docstore"]
            if hasattr(self.vector_store, 'index_to_docstore_id'):
                self.vector_store.index_to_docstore_id = store_data.get("index_to_docstore_id", {})
            
            print(f"Vector store loaded from {self.persist_dir}")
            return True
            
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return False

class AdvancedRAGSystem(LangChainRAGSystem):
    """Extended RAG system with additional features"""
    
    def __init__(self, persist_dir: str = "./vector_store"):
        super().__init__(persist_dir)
        
        # Enhanced text splitter with better parameters
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            add_start_index=True,
            separators=[
                "\n\n",  # Paragraph breaks
                "\n",    # Line breaks
                " ",     # Word breaks
                ".",     # Sentence breaks
                ",",     # Comma breaks
                ""       # Character breaks
            ]
        )
    
    def retrieve_with_filtering(self, question: str, page_filter: Optional[List[int]] = None) -> List[Document]:
        """Retrieve documents with optional page filtering"""
        # Get all similar documents
        all_docs = self.vector_store.similarity_search(question, k=10)
        
        if page_filter is not None:
            # Filter by specific pages
            filtered_docs = [
                doc for doc in all_docs 
                if doc.metadata.get("page") in page_filter
            ]
            return filtered_docs[:4]  # Return top 4 from filtered results
        
        return all_docs[:4]
    
    def enhanced_retrieve_documents(self, state: RAGState) -> RAGState:
        """Enhanced retrieval with better scoring"""
        question = state["question"]
        print(f"Enhanced retrieval for: {question}")
        
        # Use similarity search with scores
        docs_and_scores = self.vector_store.similarity_search_with_score(question, k=5)
        
        # Filter documents with similarity score > 0.7
        filtered_docs = [
            doc for doc, score in docs_and_scores 
            if score > 0.7
        ]
        
        # If no high-quality matches, take the best ones anyway
        if not filtered_docs:
            filtered_docs = [doc for doc, _ in docs_and_scores[:3]]
        
        print(f"Enhanced retrieval found {len(filtered_docs)} high-quality documents")
        
        return {"context": filtered_docs}

def main():
    """Example usage of the RAG system"""
    
    # Initialize RAG system
    rag_system = LangChainRAGSystem(persist_dir="./docs")
    
    # PDF configuration
    pdf_path = "./docs/pt_command.pdf"  # Replace with your PDF path
    
    # Try to load existing vector store first
    if not rag_system.load_vector_store():
        print("Creating new vector database from PDF...")
        success = rag_system.create_database_from_pdf(pdf_path, pages=None)  # Process all pages
        if not success:
            print("Failed to create database")
            return
        
        # Save the vector store for future use
        rag_system.save_vector_store()
    
    # Example queries
    queries = [
        "What are the main topics covered in this document?",
        "Can you summarize the key findings?",
        "What methodology was used?",
        "What are the conclusions?"
    ]
    
    print("\n" + "="*60)
    print("RAG System Ready - Processing Queries")
    print("="*60)
    
    # Process queries
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 50)
        
        try:
            # Run the query
            result = rag_system.query(query)
            
            # Display results
            print(f"Answer: {result['answer']}")
            
            # Show sources
            context_docs = result.get('context', [])
            if context_docs:
                print(f"\nSources ({len(context_docs)} documents):")
                for j, doc in enumerate(context_docs, 1):
                    page_num = doc.metadata.get('page', 'Unknown')
                    print(f"  {j}. Page {page_num + 1 if isinstance(page_num, int) else page_num}")
                    print(f"     Preview: {doc.page_content[:100]}...")
            
        except Exception as e:
            print(f"Error processing query: {e}")
        
        print()

def stream_example():
    """Example of streaming RAG responses"""
    rag_system = LangChainRAGSystem()
    
    if not rag_system.load_vector_store():
        print("No vector store found. Please run main() first to create the database.")
        return
    
    query = "What is the main purpose of this document?"
    
    print("Streaming RAG Response:")
    print("=" * 40)
    
    # Stream the workflow steps
    for step in rag_system.workflow.stream(
        {"question": query}, 
        stream_mode="updates"
    ):
        print(f"Step: {step}")

if __name__ == "__main__":
    main()
    
    # Uncomment to test streaming
    # stream_example()

# Installation requirements:
"""
pip install langchain langchain-community langchain-openai langgraph pypdf typing-extensions
"""

# Features implemented:
"""
1. PDF Loading: Uses LangChain's PyPDFLoader for robust PDF processing
2. Text Splitting: RecursiveCharacterTextSplitter with hierarchical splitting
3. Embeddings: OpenAI embeddings with custom endpoint support  
4. Vector Store: InMemoryVectorStore with similarity search
5. LLM Integration: ChatOpenAI with custom endpoint
6. LangGraph Orchestration: Structured workflow with retrieve -> generate
7. State Management: TypedDict for clean state tracking
8. Persistence: Save/load vector store functionality
9. Source Attribution: Track page numbers and provide document sources
10. Error Handling: Comprehensive error handling throughout

Advanced Features:
- Enhanced text splitting with multiple separators
- Similarity score filtering
- Page-based filtering options
- Streaming support
- Extensible architecture for customization
"""
