# GraphRAG with SurrealDB + LangChain Tutorial

This repository contains a comprehensive tutorial on building a **Graph Retrieval-Augmented Generation (GraphRAG)** chatbot using SurrealDB, LangChain, and Ollama. This implementation demonstrates how to enhance traditional RAG systems by incorporating knowledge graphs for better contextual understanding.

## 📖 What You'll Learn

This tutorial is based on the excellent SurrealDB blog post: [Make a GenAI chatbot using GraphRAG with SurrealDB + LangChain](https://surrealdb.com/blog/make-a-genai-chatbot-using-graphrag-with-surrealdb-langchain)

### Traditional RAG vs GraphRAG

**Traditional RAG** uses vector similarity search to find relevant text chunks, but can miss crucial connections and contextual information.

**GraphRAG** leverages structured knowledge graphs to provide richer, more contextualised understanding by explicitly representing entities and their relationships as nodes and edges.

### Key Benefits of GraphRAG

- **Context Enrichment**: Combines semantic meaning with structural relationships
- **Relationship Awareness**: Reveals connections that vector search alone might miss
- **Reasoning Capabilities**: Enables reasoning based on relationships
- **Improved Accuracy**: More accurate and contextually relevant answers
- **Reduced Hallucination**: Grounding responses in knowledge graphs

## 🏗️ Architecture Overview

The system follows this flow:

1. **Data Ingestion**: Load categorized health symptoms and treatments
2. **Vector Store Population**: Store documents with embeddings in SurrealDB
3. **Knowledge Graph Construction**: Build graph relationships between entities
4. **User Query Processing**: Find relevant documents via similarity search
5. **Graph Query Generation**: LLM generates graph queries to find relationships
6. **Response Generation**: Combine graph and vector results for comprehensive answers

## 📚 Notebooks Included

### 1. `graphrag_tutorial.ipynb`
**Main tutorial notebook** - Complete implementation with detailed explanations:
- Setting up SurrealDB with Docker
- Understanding vector stores and graph stores
- Data ingestion and processing
- Building knowledge graphs
- Querying with GraphRAG
- Comparing standard RAG vs GraphRAG results

### 2. `knowledge_graph_building.ipynb`
**Knowledge graph construction** - Focus on graph creation:
- Entity extraction from text
- Relationship identification
- SurrealQL query generation
- Graph visualization techniques

### 3. `embedding_setup.ipynb`
**Embedding and vector setup** - Deep dive into embeddings:
- Setting up embedding models
- Vector similarity search
- Custom embedding functions in SurrealDB

## 🛠️ Technologies Used

- **SurrealDB**: Multi-model database with native graph and vector support
- **LangChain**: Framework for LLM applications and chains
- **Ollama**: Local LLM runtime (llama3.2)
- **Python**: Main programming language
- **Docker**: For running SurrealDB locally
- **Jupyter**: Interactive development environment

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Docker and Docker Compose
- Git

### Installation

1. **Clone this repository**
```bash
git clone <repository-url>
cd graphrag-surrealdb-tutorial
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Start SurrealDB with Docker**
```bash
docker-compose up -d
```

4. **Install Ollama and pull the model**
```bash
# Install Ollama (visit https://ollama.ai for instructions)
ollama pull llama3.2
```

5. **Run the main tutorial notebook**
```bash
jupyter notebook graphrag_tutorial.ipynb
```

## 📁 Project Structure

```
graphrag-surrealdb-tutorial/
├── notebooks/
│   ├── graphrag_tutorial.ipynb          # Main tutorial
│   ├── knowledge_graph_building.ipynb   # Graph construction focus
│   └── embedding_setup.ipynb            # Vector/embedding setup
├── data/
│   ├── symptoms.yaml                     # Sample health symptoms data
│   └── example_documents/                # Additional sample documents
├── src/
│   ├── utils.py                         # Utility functions
│   ├── graph_builder.py                # Knowledge graph building
│   └── rag_chains.py                   # RAG chain implementations
├── docker-compose.yml                   # SurrealDB setup
├── requirements.txt                      # Python dependencies
├── .env.example                         # Environment variables template
└── README.md                           # This file
```

## 🔧 Configuration

1. **Copy environment template**
```bash
cp .env.example .env
```

2. **Edit `.env` file with your settings**
```env
SURREALDB_URL=ws://localhost:8000/rpc
SURREALDB_USER=root
SURREALDB_PASSWORD=root
SURREALDB_NAMESPACE=test
SURREALDB_DATABASE=test
OLLAMA_MODEL=llama3.2
```

## 📊 Sample Data

The tutorial includes a sample dataset of health symptoms and treatments structured as:

```yaml
- category: General Symptoms
  symptoms:
    - name: Fever
      description: Elevated body temperature, usually above 100.4°F (38°C)
      medical_practice: General Practice, Internal Medicine, Pediatrics
      possible_treatments:
        - Antipyretics (e.g., ibuprofen, acetaminophen)
        - Rest
        - Hydration
```

## 🎯 Key Learning Outcomes

After completing this tutorial, you'll understand:

1. **Graph Database Concepts**: How to model relationships in SurrealDB
2. **Vector Embeddings**: Creating and using embeddings for semantic search
3. **Knowledge Graph Construction**: Building graphs from unstructured text
4. **GraphRAG Implementation**: Combining vector search with graph traversal
5. **LangChain Integration**: Using chains for complex RAG workflows
6. **Query Optimization**: Efficient querying strategies for large graphs

## 🔍 Example Queries

The tutorial demonstrates various query types:

**Simple Symptom Query**:
```
"I have a runny nose and itchy eyes"
```

**Graph Query Generated**:
```sql
SELECT <-relation_Treats<-graph_Treatment as treatment
FROM graph_Symptom
WHERE name IN ["Nasal Congestion/Runny Nose", "Dizziness/Vertigo"]
```

**Enhanced Response**: GraphRAG provides more comprehensive answers by combining:
- Vector similarity results
- Graph relationship data
- Contextual entity information

## 📈 Performance Comparisons

The notebooks include detailed comparisons showing:
- **Standard RAG**: Basic vector similarity search
- **GraphRAG**: Enhanced with relationship awareness
- **Response Quality**: More detailed, contextually rich answers
- **Inference Capabilities**: Better relationship understanding

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **SurrealDB Team** for the excellent blog post and documentation
- **LangChain Community** for the powerful framework
- **Ollama Team** for making local LLMs accessible

## 🔗 Additional Resources

- [SurrealDB Documentation](https://surrealdb.com/docs)
- [LangChain Documentation](https://python.langchain.com/)
- [Ollama Models](https://ollama.ai/library)
- [Original Blog Post](https://surrealdb.com/blog/make-a-genai-chatbot-using-graphrag-with-surrealdb-langchain)

## 📞 Support

If you encounter any issues or have questions:

1. Check the notebooks for detailed explanations
2. Review the SurrealDB documentation
3. Open an issue in this repository
4. Join the SurrealDB community Discord

---

**Happy Learning!** 🚀 Build amazing GraphRAG applications with SurrealDB!
