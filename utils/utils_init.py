# utils/__init__.py
"""
NEET Buddy - Utilities Package

This package contains helper functions for the NEET Buddy application:
- PDF text extraction
- Text chunking
- Vector store management
- RAG simulation
"""

__version__ = "1.0.0"
__author__ = "NEET Buddy Development Team"

# Import main functions for easy access
from .helper_functions import (
    get_pdf_text,
    get_text_chunks, 
    get_vector_store,
    get_conversational_chain_placeholder,
    load_vector_store,
    get_document_stats
)

__all__ = [
    "get_pdf_text",
    "get_text_chunks",
    "get_vector_store", 
    "get_conversational_chain_placeholder",
    "load_vector_store",
    "get_document_stats"
]