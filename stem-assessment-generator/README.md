# STEM Assessment Generator

A FastAPI-powered web application that automatically generates high-quality assessment questions from educational PDF documents using AI and RAG (Retrieval Augmented Generation) technology.

## Features

- 📄 **PDF Processing**: Upload and parse educational PDF documents
- 🤖 **AI-Powered Generation**: Generate questions using OpenAI's GPT models
- 🎯 **Smart Distractors**: Create plausible incorrect answers based on common student misconceptions
- 🔍 **RAG Technology**: Use vector embeddings and semantic search for context-aware question generation
- 📊 **Multiple Difficulty Levels**: Easy, Medium, and Hard question difficulty
- 🎨 **Modern Web Interface**: Clean, responsive UI with drag-and-drop file upload
- ⚙️ **Configurable**: Customizable question count, difficulty, and other parameters

## Technology Stack

- **Backend**: FastAPI, Python 3.8+
- **AI/ML**: OpenAI GPT-3.5/4, LangChain, ChromaDB
- **PDF Processing**: PyPDF2
- **Vector Database**: ChromaDB for embeddings storage
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Deployment**: Uvicorn ASGI server

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Git

### Setup

1. **Clone the repository**
   ```bash
   cd /path/to/your/workspace
   git clone <repository-url>
   cd stem-assessment-generator
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env file and add your OpenAI API key
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the application**
   - Open your browser and go to `http://localhost:8000`

## Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```env
# OpenAI Configuration
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# App Configuration  
APP_HOST=0.0.0.0
APP_PORT=8000
DEBUG=True

# File Configuration
MAX_FILE_SIZE_MB=10
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

### Customization

- **Question Types**: Currently supports multiple choice questions
- **Difficulty Levels**: Easy, Medium, Hard
- **Misconception Patterns**: Defined in `patterns.py` for various STEM subjects
- **Chunk Size**: Adjustable text chunking for RAG pipeline

## Usage

1. **Upload PDF**: Drag and drop or click to upload an educational PDF document
2. **Configure Settings**: Choose number of questions (3-10) and difficulty level
3. **Generate Assessment**: Click "Generate Assessment Questions" 
4. **Review Questions**: Interactive interface to view questions and check answers
5. **Start Over**: Generate new assessments from different documents

## Project Structure

```
stem-assessment-generator/
├── app.py              # Main FastAPI application
├── pdf_processor.py    # PDF parsing and chunking
├── rag_pipeline.py     # Embeddings, vector store, retrieval
├── generator.py        # Question and distractor generation
├── patterns.py         # All misconception patterns
├── models.py           # Pydantic models
├── config.py           # Configuration
├── static/
│   ├── index.html     # Frontend HTML
│   └── app.js         # Frontend JavaScript
├── data/
│   ├── uploads/       # Temporary PDF storage
│   └── vectordb/      # ChromaDB persistence
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## API Endpoints

### `GET /`
- Serves the main web interface

### `POST /upload`
- Uploads PDF and generates assessment questions
- **Form Data**:
  - `file`: PDF file (max 10MB)
  - `num_questions`: Number of questions (3-10)
  - `difficulty`: Difficulty level (easy/medium/hard)
- **Response**: JSON with generated questions and metadata

## Misconception Patterns

The system includes predefined misconception patterns for:

- **Physics**: Force and motion, energy conservation
- **Chemistry**: Atomic structure, chemical bonding  
- **Biology**: Evolution, genetics
- **Mathematics**: Algebra, fractions

These patterns help generate realistic distractors that reflect common student errors.

## Development

### Adding New Misconception Patterns

Edit `patterns.py` to add new subjects or topics:

```python
MISCONCEPTION_PATTERNS["new_subject"] = {
    "topic_name": MisconceptionPattern(
        name="Pattern Name",
        description="Description of misconception",
        examples=["Example 1", "Example 2"],
        distractor_templates=["Template 1", "Template 2"]
    )
}
```

### Testing

```bash
# Run basic tests
python -c "import app; print('Import successful')"

# Test PDF processing
python -c "from pdf_processor import PDFProcessor; print('PDF processor ready')"

# Test RAG pipeline
python -c "from rag_pipeline import RAGPipeline; print('RAG pipeline ready')"
```

## Troubleshooting

### Common Issues

1. **OpenAI API Key Error**
   - Ensure your API key is correctly set in `.env`
   - Check API key permissions and credits

2. **PDF Processing Fails**
   - Verify PDF is not password protected
   - Check file size is under 10MB
   - Ensure PDF contains extractable text

3. **ChromaDB Issues**
   - Delete `data/vectordb/` folder and restart
   - Check disk space availability

4. **Memory Issues**
   - Reduce `CHUNK_SIZE` in configuration
   - Process smaller PDF documents

### Performance Optimization

- Use SSD storage for `data/vectordb/` 
- Increase `CHUNK_SIZE` for longer documents
- Consider using GPT-4 for better quality (higher cost)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for GPT models and embeddings
- LangChain for RAG framework
- ChromaDB for vector storage
- FastAPI for the web framework

## Support

For questions or issues:
1. Check the troubleshooting section
2. Review the GitHub issues
3. Create a new issue with detailed information

---

**Note**: This application requires an OpenAI API key and will incur usage costs based on your API usage. Monitor your usage through the OpenAI dashboard.
