# NEET Buddy - Your AI Study Partner 📚

NEET Buddy is an AI-powered educational application designed specifically for NEET aspirants. Upload your study materials and ask questions to get instant, contextual answers powered by advanced Retrieval-Augmented Generation (RAG) technology - all running locally on your machine!

## 🌟 Features

- **📄 PDF Upload & Processing**: Upload multiple PDF study materials and extract text content automatically
- **🧠 Smart Text Chunking**: Intelligently splits large documents into manageable, searchable chunks
- **🔍 Semantic Search**: Uses advanced embeddings to find the most relevant content for your questions
- **💬 Interactive Chat Interface**: Ask questions and get contextual answers based on your uploaded materials
- **💾 Local Storage**: All processing happens locally - your data never leaves your machine
- **⚡ Fast Retrieval**: FAISS-powered vector search for lightning-fast information retrieval
- **🎯 NEET-Focused**: Designed specifically for medical entrance exam preparation

## 🛠️ Technology Stack

- **Web Framework**: Streamlit
- **PDF Processing**: PyMuPDF (fitz)
- **Text Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Language Model**: Placeholder implementation (ready for integration)

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)

## 🚀 How to Run Locally

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/NEET_Buddy_Streamlit.git
cd NEET_Buddy_Streamlit
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
streamlit run app.py
```

### Step 5: Open Your Browser
The application will automatically open in your default browser at `http://localhost:8501`

## 📖 How to Use

1. **Upload Documents**: 
   - Use the sidebar to upload one or more PDF study materials
   - Click "Process Documents" to extract and index the content

2. **Ask Questions**:
   - Once processing is complete, use the chat interface to ask questions
   - The system will search your uploaded materials and provide relevant answers

3. **Manage Your Knowledge Base**:
   - Load previously processed documents using "Load Existing Index"
   - Add new documents by uploading additional PDFs

## 📁 Project Structure

```
NEET_Buddy_Streamlit/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore file
├── README.md             # Project documentation
└── utils/
    ├── __init__.py       # Package initialization
    └── helper_functions.py # Core backend logic
```

## 🔧 Configuration

The application uses the following default settings:
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters
- **Embedding Model**: all-MiniLM-L6-v2
- **Search Results**: Top 3 most relevant chunks

These can be modified in the `helper_functions.py` file.

## 🚧 Current Limitations

- **Placeholder LLM**: Currently uses a dummy response system. Ready for integration with actual language models like Ollama, GPT, or local models.
- **PDF Only**: Currently supports PDF files only
- **English Language**: Optimized for English text content

## 🛣️ Future Enhancements

- [ ] Integration with local LLM models (Ollama, LLaMA, etc.)
- [ ] Support for additional file formats (Word, PPT, HTML)
- [ ] Multi-language support
- [ ] Advanced filtering and search options
- [ ] Study progress tracking
- [ ] Question history and bookmarking

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Sentence-Transformers team for the embedding models
- Facebook AI Research for FAISS
- Streamlit team for the amazing web framework
- PyMuPDF developers for PDF processing capabilities

## 📞 Support

If you encounter any issues or have questions:
1. Check the existing issues in the repository
2. Create a new issue with a detailed description
3. Include error messages and system information

---

**Happy Studying! 🎓✨**

*Built with ❤️ for NEET aspirants*