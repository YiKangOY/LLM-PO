"""
Simple RAG utilities for ECO optimization knowledge retrieval
Combines optimization strategies and unfixable reason explanations
"""

import os
from typing import List, Dict, Any
from pathlib import Path

# LangChain components
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

def _openai_kwargs() -> Dict[str, str]:
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    kwargs: Dict[str, str] = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    return kwargs

class ECOKnowledgeRAG:
    """Simple RAG system for ECO optimization knowledge"""
    
    def __init__(self, planning_files=None):
        if planning_files is None:
            planning_files = ["opt_target_strategies.txt"]
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            **_openai_kwargs()
        )
        self.vector_store = InMemoryVectorStore(self.embeddings)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,      # Large enough to keep entire sections together
            chunk_overlap=0,     # No overlap - sections are semantically distinct
            separators=["---"]   # Split chunks strictly on '---' section dividers
        )
        self.knowledge_loaded = False
        self.planning_files = planning_files
        self.corpus_fingerprint = []
        self.docs_path = Path(__file__).parent.parent / "docs"
        self.default_vector_store_path = str(self.docs_path / f"{Path(self.planning_files[0]).stem}.pkl")

    def _planning_doc_list(self) -> List[str]:
        return self.planning_files

    def _compute_corpus_fingerprint(self, docs_path: Path, filenames: List[str]) -> List[Dict[str, Any]]:
        fingerprint = []
        for filename in filenames:
            file_path = docs_path / filename
            fingerprint.append({
                "file": filename,
                "modified_time": int(file_path.stat().st_mtime)
            })
        return fingerprint
    
    def build_knowledge_database(self):
        """Build vector database from planning documents"""
        docs_path = self.docs_path
        documents = []

        planning_files = self._planning_doc_list()

        for filename in planning_files:
            file_path = docs_path / filename
            if not file_path.exists():
                raise FileNotFoundError(f"Planning document not found: {file_path}")

            with open(file_path, "r") as f:
                content = f.read()

            if not content.strip():
                raise ValueError(f"Planning document is empty: {file_path}")

            document = Document(
                page_content=content,
                metadata={
                    "source": file_path.stem,
                    "type": "task_planning"
                }
            )
            documents.append(document)
        
        # Split documents into chunks
        all_chunks = self.text_splitter.split_documents(documents)

        if not all_chunks:
            raise ValueError("No chunks generated from documents")

        # Filter and clean chunks
        filtered_chunks = []
        for chunk in all_chunks:
            content = chunk.page_content.strip()
            # Skip separator-only chunks
            if not content or content == "---":
                continue
            # Remove leading separator if present
            if content.startswith("---"):
                content = content[3:].strip()
            # Update chunk content
            chunk.page_content = content
            filtered_chunks.append(chunk)

        if not filtered_chunks:
            raise ValueError("No valid chunks after filtering")

        print(f"Created {len(filtered_chunks)} knowledge chunks (filtered from {len(all_chunks)})")

        # Add to vector store
        self.vector_store.add_documents(filtered_chunks)
        print(f"Built knowledge database with {len(filtered_chunks)} chunks")

        self.planning_files = planning_files
        self.corpus_fingerprint = self._compute_corpus_fingerprint(docs_path, planning_files)
        self.knowledge_loaded = True
    
    def query_knowledge(self, queries: List[str], k: int = 1) -> List[Dict[str, Any]]:
        """Simple query interface for knowledge retrieval"""
        if not self.knowledge_loaded:
            raise RuntimeError("Knowledge database not loaded. Call build_knowledge_database() first.")
        
        if not queries:
            raise ValueError("No queries provided for knowledge retrieval")
        
        all_results = []
        
        for query in queries:
            if not query.strip():
                raise ValueError(f"Empty query provided: '{query}'")
            
            # Perform similarity search
            docs = self.vector_store.similarity_search(query, k=k)
            
            if not docs:
                raise RuntimeError(f"No documents retrieved for query: '{query}'")
            
            # Format results
            query_results = {
                "query": query,
                "retrieved_docs": [
                    {
                        "content": doc.page_content,
                        "source": doc.metadata["source"],
                        "type": doc.metadata["type"]
                    }
                    for doc in docs
                ]
            }
            all_results.append(query_results)
        
        return all_results
    
    def save_database(self, path: str = None):
        """Save vector database for reuse"""
        import pickle

        if path is None:
            path = self.default_vector_store_path

        if not self.knowledge_loaded:
            raise RuntimeError("Cannot save database - knowledge not loaded")

        # Ensure directory exists
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        store_data = {
            "docstore": self.vector_store.store,
            "knowledge_loaded": self.knowledge_loaded,
            "planning_files": self.planning_files,
            "corpus_fingerprint": self.corpus_fingerprint
        }
        
        with open(path, "wb") as f:
            pickle.dump(store_data, f)
        
        print(f"Knowledge database saved to {path}")
    
    def load_database(self, path: str = None) -> bool:
        """Load existing vector database"""
        import pickle

        if path is None:
            path = self.default_vector_store_path

        if not os.path.exists(path):
            return False
        
        with open(path, "rb") as f:
            store_data = pickle.load(f)
        
        if "docstore" not in store_data:
            raise ValueError(f"Corrupted database file: missing docstore in {path}")

        if "planning_files" not in store_data or "corpus_fingerprint" not in store_data:
            return False
        
        # Recreate vector store
        self.vector_store = InMemoryVectorStore(self.embeddings)
        self.vector_store.store = store_data["docstore"]
        self.planning_files = store_data["planning_files"]
        saved_fingerprint = store_data["corpus_fingerprint"]

        docs_path = self.docs_path
        current_fingerprint = self._compute_corpus_fingerprint(docs_path, self.planning_files)

        if current_fingerprint != saved_fingerprint:
            print("Planning documents changed since last build. Rebuilding RAG database...")
            return False

        self.knowledge_loaded = store_data["knowledge_loaded"]
        
        if not self.knowledge_loaded:
            raise ValueError(f"Database indicates knowledge not loaded: {path}")
        
        print(f"Knowledge database loaded from {path}")
        return True

def create_eco_rag_system(source_file: str = "general_opt_strategies.txt", vector_store_path: str = None) -> ECOKnowledgeRAG:
    """Factory function to create and initialize RAG system"""
    rag_system = ECOKnowledgeRAG(planning_files=[source_file])
    
    # Try to load existing database first
    if not rag_system.load_database(path=vector_store_path):
        print("Building new knowledge database...")
        rag_system.build_knowledge_database()
        rag_system.save_database(path=vector_store_path)
    
    return rag_system

# Test function
if __name__ == "__main__":
    # Test the RAG system
    rag = create_eco_rag_system()
    
    # Test queries
    test_queries = [
        """# Timing Fixing Guidelines
1. When strategy is Exploration, broadly explore different command options combinations to cover more possibilities and reveal unfixable reasons for future optimization iterations.
2. When strategy is Exploitation, understand the optimization history and unfixable reasons and their definitions to target the potential fixing options. Also avoid repeating the previous optimization attempts that has no effect. For example, if many unfixable reason says the violation is in the clock tree, then choose cell_class as clock_tree.
3. You should treat fixing setup and hold violations seperately, with seperate exploration and exploitation progress control.
4. If the unfixable reasons are non fixable, ignore them since you cannot fix them.
"""
    ]
    
    results = rag.query_knowledge(test_queries)
    
    for result in results:
        print(f"\nQuery: {result['query']}")
        print("Retrieved content:")
        for i, doc in enumerate(result['retrieved_docs'], 1):
            print(f"  {i}. [{doc['source']}] {doc['content']}...")
