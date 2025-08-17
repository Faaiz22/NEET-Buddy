try:
    import fitz  # PyMuPDF
except ImportError:
    try:
        import PyMuPDF as fitz
    except ImportError:
        st.error("PyMuPDF not installed. Please install it using: pip install PyMuPDF")
        st.stop()

import os
import pickle
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Optional
import streamlit as st

# Initialize the embedding model globally to avoid reloading
@st.cache_resource
def load_embedding_model():
    """Load and cache the sentence transformer model"""
    return SentenceTransformer('all-MiniLM-L6-v2')

def get_pdf_text(pdf_docs) -> str:
    """
    Extract text from uploaded PDF files using PyMuPDF
    
    Args:
        pdf_docs: List of uploaded PDF files from Streamlit
    
    Returns:
        str: Combined text from all PDF files
    """
    text = ""
    
    for pdf in pdf_docs:
        try:
            # Read the PDF file
            pdf_document = fitz.open(stream=pdf.read(), filetype="pdf")
            
            # Extract text from each page
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                page_text = page.get_text()
                text += page_text
            
            pdf_document.close()
            
        except Exception as e:
            st.error(f"Error processing PDF {pdf.name}: {str(e)}")
            continue
    
    return text

def get_text_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks
    
    Args:
        text (str): Raw text to split
        chunk_size (int): Size of each chunk
        chunk_overlap (int): Overlap between consecutive chunks
    
    Returns:
        List[str]: List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        # Calculate end position
        end = start + chunk_size
        
        # If this is not the last chunk, try to end at a sentence boundary
        if end < len(text):
            # Look for sentence endings within the last 100 characters
            last_period = text.rfind('.', end - 100, end)
            last_question = text.rfind('?', end - 100, end)
            last_exclamation = text.rfind('!', end - 100, end)
            
            # Use the latest sentence ending found
            sentence_end = max(last_period, last_question, last_exclamation)
            if sentence_end > start:
                end = sentence_end + 1
        
        # Extract the chunk
        chunk = text[start:end].strip()
        
        # Only add non-empty chunks
        if chunk:
            chunks.append(chunk)
        
        # Move start position considering overlap
        if end >= len(text):
            break
        start = end - chunk_overlap
    
    return chunks

def get_vector_store(text_chunks: List[str]) -> faiss.IndexFlatL2:
    """
    Create FAISS vector store from text chunks
    
    Args:
        text_chunks (List[str]): List of text chunks
    
    Returns:
        faiss.IndexFlatL2: FAISS index with embeddings
    """
    # Load the embedding model
    embedding_model = load_embedding_model()
    
    # Generate embeddings for all chunks
    embeddings = embedding_model.encode(text_chunks, convert_to_tensor=False)
    embeddings = np.array(embeddings).astype('float32')
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    
    # Add embeddings to index
    index.add(embeddings)
    
    # Save the index and chunks to local files
    try:
        faiss.write_index(index, "faiss_index")
        
        # Save the text chunks separately
        with open("text_chunks.pkl", "wb") as f:
            pickle.dump(text_chunks, f)
        
        st.success(f"Vector store saved with {len(text_chunks)} chunks!")
        
    except Exception as e:
        st.error(f"Error saving vector store: {str(e)}")
    
    # Store chunks in the index object for retrieval
    index.text_chunks = text_chunks
    
    return index

def load_vector_store() -> Optional[faiss.IndexFlatL2]:
    """
    Load existing FAISS vector store from local files
    
    Returns:
        Optional[faiss.IndexFlatL2]: Loaded FAISS index or None if not found
    """
    try:
        if os.path.exists("faiss_index") and os.path.exists("text_chunks.pkl"):
            # Load the FAISS index
            index = faiss.read_index("faiss_index")
            
            # Load the text chunks
            with open("text_chunks.pkl", "rb") as f:
                text_chunks = pickle.load(f)
            
            # Attach chunks to index for retrieval
            index.text_chunks = text_chunks
            
            return index
        else:
            return None
    
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None

def get_conversational_chain_placeholder(vector_store: faiss.IndexFlatL2, query: str, k: int = 3) -> str:
    """
    Placeholder function that simulates RAG process
    
    Args:
        vector_store: FAISS vector store
        query (str): User's question
        k (int): Number of top results to retrieve
    
    Returns:
        str: Dummy answer with retrieved context
    """
    try:
        # Load embedding model
        embedding_model = load_embedding_model()
        
        # Generate embedding for the query
        query_embedding = embedding_model.encode([query], convert_to_tensor=False)
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Perform similarity search
        scores, indices = vector_store.search(query_embedding, k)
        
        # Retrieve the top k text chunks
        retrieved_chunks = []
        for i, idx in enumerate(indices[0]):
            if idx < len(vector_store.text_chunks):
                chunk = vector_store.text_chunks[idx]
                score = scores[0][i]
                retrieved_chunks.append({
                    'text': chunk[:200] + "..." if len(chunk) > 200 else chunk,
                    'score': float(score)
                })
        
        # Format the response
        if retrieved_chunks:
            context_text = "\n\n".join([f"**Context {i+1}** (similarity: {chunk['score']:.3f}):\n{chunk['text']}" 
                                       for i, chunk in enumerate(retrieved_chunks)])
            
            response = f"""
**🤖 DUMMY ANSWER**: Based on the provided context, here's what I found regarding your question: *"{query}"*

**📝 Retrieved Context:**
{context_text}

**💡 Analysis:**
Based on the retrieved information, the answer to your question is likely related to the key points mentioned above. This is a placeholder response - in a full implementation, a language model would analyze these contexts and provide a comprehensive answer.

**📊 Search Results:** Found {len(retrieved_chunks)} relevant text chunks from your study materials.
            """
        else:
            response = f"""
**🤖 DUMMY ANSWER**: I searched for information related to your question: *"{query}"*

Unfortunately, I couldn't find highly relevant content in your uploaded study materials. This could mean:
- The topic might not be covered in the uploaded documents
- Try rephrasing your question with different keywords
- The content might be there but with different terminology

**💡 Suggestion:** Try asking more specific questions or upload additional study materials that cover this topic.
            """
        
        return response
    
    except Exception as e:
        return f"**❌ Error**: I encountered an issue while searching for your answer: {str(e)}"

def get_document_stats() -> dict:
    """
    Get statistics about the current vector store
    
    Returns:
        dict: Statistics about the loaded documents
    """
    try:
        if os.path.exists("text_chunks.pkl"):
            with open("text_chunks.pkl", "rb") as f:
                text_chunks = pickle.load(f)
            
            total_chunks = len(text_chunks)
            total_chars = sum(len(chunk) for chunk in text_chunks)
            avg_chunk_size = total_chars // total_chunks if total_chunks > 0 else 0
            
            return {
                'total_chunks': total_chunks,
                'total_characters': total_chars,
                'avg_chunk_size': avg_chunk_size
            }
        else:
            return {'total_chunks': 0, 'total_characters': 0, 'avg_chunk_size': 0}
    
    except Exception:
        return {'total_chunks': 0, 'total_characters': 0, 'avg_chunk_size': 0}
