# AI File Search and Document query Chatbot

An AI-powered document interaction system built using Streamlit, LlamaIndex, Hugging Face embeddings, and Mistral-7B.

The chatbot enables users to upload files, generate vector embeddings, and ask natural language questions about document content. The system uses Retrieval-Augmented Generation (RAG) to provide context-aware responses from uploaded files.

---

## Features

- Upload multiple documents
- AI-powered document querying
- Retrieval-Augmented Generation (RAG)
- Vector-based semantic search
- Persistent vector database storage
- Chat history support
- Hugging Face inference integration
- Streamlit interactive interface

---

## Tech Stack

- Python
- Streamlit
- LlamaIndex
- Hugging Face API
- Mistral-7B
- Hugging Face Embeddings

---

## Project Workflow

1. User uploads one or more documents.
2. The system:
   - Reads document content
   - Generates embeddings
   - Creates a vector index

3. The vector database is stored locally.

4. User enters queries in natural language.

5. The retriever fetches relevant context from documents.

6. Mistral-7B generates contextual responses using retrieved information.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ai-document-chatbot.git
cd ai-document-chatbot
