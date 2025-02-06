# RAG System with Custom Knowledge

## Prerequisites

- Docker
- Docker Compose

## Setup

1. Clone the repository

```bash
git clone https://github.com/nengapi/rag-system-custom-knowledge.git
```

2. Build the Docker image

```bash
docker compose build
```

3. Run the Docker container

```bash
docker compose up -d
```

4. Test the RAG system

```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question": "What general characteristics of unicorn?"}'
```

## Tips

- If you want to change the knowledge base, you can edit the `data/knowledge.txt` file.
- If you want to change the LLM model, you can edit the `entrypoint.sh` file.