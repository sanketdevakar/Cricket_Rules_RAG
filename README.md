# Cricket Laws RAG Application

## Overview
A Conversational Retrieval-Augmented Generation (RAG) system that provides intelligent responses about cricket laws by combining vector search and large language models.

## Architecture
- **Graph-based Workflow**: Built using LangGraph's StateGraph for modular processing
- **Vector Database**: Milvus/Zilliz Cloud for semantic search
- **LLM Integration**: Groq API for natural language generation
- **Conversation Management**: Stateful system maintaining chat context

## Components
### 1. Retrieval System
- Semantic search using Milvus vector database
- Sentence transformer embeddings (BAAI/bge-large-en)
- Top-k retrieval of relevant cricket laws

### 2. Grading Module
- Dynamic relevance scoring for retrieved chunks
- Automated content filtering based on relevance
- Individual chunk evaluation for precision

### 3. LLM Integration
- Groq API integration for response generation
- Streaming responses for better user experience
- Context-aware prompt engineering

### 4. Conversation Management
- Chat history tracking
- Follow-up question handling
- Context persistence across interactions

## Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Environment variables:
```env
GROQ_API_KEY=your_groq_api_key
ZILLIZ_URI=your_zilliz_uri
ZILLIZ_TOKEN=your_zilliz_token
COLLECTION_NAME=cricket_rules_subchunks
```

3. Run the application:
```bash
python main.py
```

## Usage
```bash
> Question: What are the rules for a wide ball?
> Follow-up: What happens if a batsman hits a wide ball?
```

## Project Structure
```
Cricket_RAG/
├── graph/
│   ├── rag_graph.py    # Graph architecture
│   └── state.py        # State management
├── retriever/
│   └── milvus_retriever.py  # Vector search
├── main.py             # CLI interface
└── README.md
```

## Features
- Real-time streaming responses
- Context-aware conversations
- Source citations for answers
- Modular graph-based architecture
- Semantic search capabilities

## Dependencies
- LangGraph
- Pymilvus
- Sentence Transformers
- Python-dotenv
- Requests