# Docater

Docater is a powerful document analysis application that leverages local **Ollama** models to create an interactive chat interface for PDF documents.

## Features

- PDF document processing and analysis
- Local LLM integration using Ollama models
- Interactive chat interface for document queries
- GPU acceleration support (CUDA)
- Document processing caching
- Multi-document management

## Prerequisites

- Python 3.8+
- Ollama installed and running
- CUDA-compatible GPU (optional)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/martinsagabriel/Docater.git
cd Docater
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure Ollama is running with required models installed:
```bash
ollama pull llama2
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided URL (typically http://localhost:8501)

3. Upload a PDF document and start chatting!

## Features in Detail

- **Document Processing**: Upload and process PDF documents
- **Intelligent Caching**: Processed documents are cached for faster subsequent access
- **Model Selection**: Uses pre-configured Ollama models for optimal performance
- **Chat Interface**: Natural conversation interface for document queries
- **GPU Acceleration**: Automatic CUDA GPU detection and utilization
- **Database Management**: Easy cleanup of processed documents

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.