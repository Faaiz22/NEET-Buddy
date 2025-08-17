import streamlit as st
import os
from utils.helper_functions import (
    get_pdf_text,
    get_text_chunks,
    get_vector_store,
    get_conversational_chain_placeholder,
    load_vector_store
)

def main():
    st.set_page_config(page_title="NEET Buddy", page_icon="📚", layout="wide")
    
    st.title("NEET Buddy - Your AI Study Partner 📚")
    st.markdown("Upload your study materials and ask questions to boost your NEET preparation!")
    
    # Initialize session state
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar for document upload and processing
    with st.sidebar:
        st.header("📖 Document Management")
        
        # File uploader
        pdf_docs = st.file_uploader(
            "Upload your PDF study materials",
            type=["pdf"],
            accept_multiple_files=True,
            help="Select one or more PDF files containing your study material"
        )
        
        # Process documents button
        if st.button("🔄 Process Documents", type="primary"):
            if pdf_docs:
                with st.spinner("Processing PDFs... This may take a few moments."):
                    try:
                        # Extract text from PDFs
                        raw_text = get_pdf_text(pdf_docs)
                        
                        if not raw_text.strip():
                            st.error("No text could be extracted from the uploaded PDFs. Please check if they contain readable text.")
                            return
                        
                        # Split text into chunks
                        text_chunks = get_text_chunks(raw_text)
                        
                        # Create and save vector store
                        vector_store = get_vector_store(text_chunks)
                        st.session_state.vector_store = vector_store
                        
                        st.success(f"✅ Documents processed successfully! Extracted {len(text_chunks)} text chunks. You can now ask questions.")
                        
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
            else:
                st.warning("Please upload at least one PDF file before processing.")
        
        # Load existing vector store if available
        if st.button("📁 Load Existing Index"):
            try:
                vector_store = load_vector_store()
                if vector_store:
                    st.session_state.vector_store = vector_store
                    st.success("✅ Existing vector store loaded successfully!")
                else:
                    st.info("No existing vector store found. Please upload and process documents first.")
            except Exception as e:
                st.error(f"Error loading vector store: {str(e)}")
        
        # Display status
        st.markdown("---")
        if st.session_state.vector_store:
            st.success("📊 Ready to answer questions!")
        else:
            st.info("📤 Please upload and process documents to get started.")
    
    # Main chat interface
    st.header("💬 Ask Your Questions")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if question := st.chat_input("Ask a question about your study material..."):
        if st.session_state.vector_store:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = get_conversational_chain_placeholder(
                            st.session_state.vector_store, 
                            question
                        )
                        st.markdown(response)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    except Exception as e:
                        error_msg = f"Sorry, I encountered an error while processing your question: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            st.warning("⚠️ Please upload and process documents first before asking questions!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <small>NEET Buddy v1.0 - Built with Streamlit | For educational purposes only</small>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
