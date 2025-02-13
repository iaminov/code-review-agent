# Context-Aware Code Reviewer

An autonomous code review assistant that uses a RAG pipeline with the OpenAI API to provide context-aware feedback and suggestions.

## Architecture

The application is built around a Retrieval-Augmented Generation (RAG) pipeline that leverages a vector store for context retrieval and a large language model for generating code reviews.

- **FastAPI Backend**: Serves the application with two main endpoints:
  - `/upload`: Ingests and processes code files.
  - `/review`: Generates a review for a given code snippet.
- **Ingestor**: Reads files, splits them into manageable chunks, and generates embeddings using OpenAI's API.
- **Vector Store (FAISS)**: Stores the embeddings and allows for efficient similarity searches.
- **RAG Chain (LangChain)**: Constructs a prompt with retrieved context and the code to be reviewed, then queries the OpenAI chat model to generate feedback.
- **Docker**: Containerizes the application for consistent deployment.
- **GitHub Actions**: Provides CI for linting, testing, and building.

## Getting Started

### Prerequisites

- Docker
- Python 3.10
- An OpenAI API key

### Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/iaminov/code-review-agent.git
    cd code-review-agent
    ```

2.  **Set up the environment:**

    Create a `.env` file in the root directory and add your OpenAI API key:
    ```
    OPENAI_API_KEY="your_api_key_here"
    ```

3.  **Build and run the Docker container:**
    ```bash
    docker build -t code-reviewer .
    docker run -p 8000:8000 --env-file .env code-reviewer
    ```

    The application will be available at `http://localhost:8000`.

## API Usage

### 1. Upload Files for Context

Upload a ZIP file containing the source code to the `/upload` endpoint. This will populate the vector store with context for the review.

- **URL**: `/upload`
- **Method**: `POST`
- **Body**: `file` (form-data)

**Example using `curl`:**
```bash
curl -X POST -F "file=@/path/to/your/code.zip" http://localhost:8000/upload
```

### 2. Request a Code Review

Submit a code snippet to the `/review` endpoint to get feedback. The RAG chain will use the uploaded context to provide a relevant review.

- **URL**: `/review`
- **Method**: `POST`
- **Body**: (JSON)
  ```json
  {
    "code": "your_code_snippet_here"
  }
  ```

**Example using `curl`:**
```bash
curl -X POST -H "Content-Type: application/json" \
-d '{"code": "def example_function():\n    return 1"}' \
http://localhost:8000/review
```

This completes all the planned commits for the project. The repository is now fully functional, with a complete CI/CD pipeline and comprehensive documentation.
