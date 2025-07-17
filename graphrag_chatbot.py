#!/usr/bin/env python3
"""
GraphRAG Chatbot Implementation using SurrealDB and LangChain

This script demonstrates how to build a GenAI chatbot using GraphRAG (Graph-Enhanced 
Retrieval-Augmented Generation) with SurrealDB as both the vector store and graph store.

Based on the SurrealDB blog post:
https://surrealdb.com/blog/make-a-genai-chatbot-using-graphrag-with-surrealdb-langchain

Key Features:
- Vector similarity search using SurrealDB
- Knowledge graph traversal for relationship-aware retrieval
- Combination of traditional RAG with graph-based reasoning
- Support for symptoms/treatments medical knowledge base example

Requirements:
- SurrealDB running locally or remotely
- OpenAI API key or Ollama for local LLMs
- Python dependencies managed via UV
"""

import asyncio
import os
import yaml
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import click

# Core dependencies
from surrealdb import Surreal
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.graphs import SurrealDBGraph
from langchain_community.vectorstores import SurrealDBVectorStore
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.graphs.graph_document import GraphDocument, Node, Relationship
from langchain.chains.graph_qa.surrealdb import SurrealDBGraphQAChain

# Alternative: OpenAI instead of Ollama
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


@dataclass
class Symptom:
    """Data structure for representing medical symptoms"""
    name: str
    description: str
    medical_practice: str
    possible_treatments: List[str]


@dataclass 
class Symptoms:
    """Container for categorized symptoms"""
    category: str
    symptoms: List[Symptom]


class GraphRAGChatbot:
    """
    GraphRAG Chatbot implementation using SurrealDB as both vector and graph store
    
    This chatbot combines traditional vector similarity search with knowledge graph
    traversal to provide more contextually aware and relationship-rich responses.
    """
    
    def __init__(
        self, 
        surrealdb_url: str = "ws://localhost:8000/rpc",
        namespace: str = "test",
        database: str = "test",
        username: str = "root", 
        password: str = "root",
        use_openai: bool = False,
        openai_api_key: Optional[str] = None,
        model_name: str = "llama3.2",
        verbose: bool = False
    ):
        """
        Initialize the GraphRAG chatbot
        
        Args:
            surrealdb_url: SurrealDB connection URL
            namespace: SurrealDB namespace
            database: SurrealDB database name
            username: SurrealDB username
            password: SurrealDB password
            use_openai: Whether to use OpenAI instead of Ollama
            openai_api_key: OpenAI API key (if using OpenAI)
            model_name: Name of the LLM model to use
            verbose: Enable verbose logging
        """
        self.surrealdb_url = surrealdb_url
        self.namespace = namespace
        self.database = database
        self.username = username
        self.password = password
        self.verbose = verbose
        
        # Initialize SurrealDB connection
        self.conn = None
        self.vector_store = None
        self.graph_store = None
        self.chat_model = None
        self.embeddings = None
        
        # Configure LLM and embeddings
        if use_openai and OPENAI_AVAILABLE and openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            self.chat_model = ChatOpenAI(model=model_name, temperature=0)
            self.embeddings = OpenAIEmbeddings()
            click.echo("Using OpenAI models")
        else:
            self.chat_model = ChatOllama(model=model_name, temperature=0)
            self.embeddings = OllamaEmbeddings(model=model_name)
            click.echo(f"Using Ollama with model: {model_name}")
    
    async def initialize_connection(self):
        """Initialize connection to SurrealDB and set up stores"""
        try:
            # Connect to SurrealDB
            self.conn = Surreal(self.surrealdb_url)
            await self.conn.signin({"username": self.username, "password": self.password})
            await self.conn.use(self.namespace, self.database)
            
            # Initialize vector store
            self.vector_store = SurrealDBVectorStore(
                embedding=self.embeddings,
                client=self.conn,
                table_name="documents"
            )
            
            # Initialize graph store
            self.graph_store = SurrealDBGraph(client=self.conn)
            
            click.echo("✅ Successfully connected to SurrealDB and initialized stores")
            
        except Exception as e:
            click.echo(f"❌ Error connecting to SurrealDB: {e}")
            raise
    
    def load_sample_data(self, data_path: Optional[str] = None) -> List[Document]:
        """
        Load sample symptoms data from YAML file or create default data
        
        Args:
            data_path: Path to YAML data file (optional)
            
        Returns:
            List of Document objects ready for ingestion
        """
        if data_path and os.path.exists(data_path):
            # Load from YAML file
            with open(data_path, "r") as f:
                symptoms_data = yaml.safe_load(f)
        else:
            # Create sample data structure based on the blog post example
            symptoms_data = [
                {
                    "category": "General Symptoms",
                    "symptoms": [
                        {
                            "name": "Fever",
                            "description": "Elevated body temperature, usually above 100.4°F (38°C).",
                            "medical_practice": "General Practice, Internal Medicine, Pediatrics",
                            "possible_treatments": [
                                "Antipyretics (e.g., ibuprofen, acetaminophen)",
                                "Rest",
                                "Hydration",
                                "Treating the underlying cause"
                            ]
                        },
                        {
                            "name": "Nasal Congestion/Runny Nose",
                            "description": "Stuffy nose due to inflamed nasal passages or a dripping nose with mucus discharge.",
                            "medical_practice": "ENT, General Practice, Allergy & Immunology",
                            "possible_treatments": [
                                "Decongestants (oral or nasal sprays)",
                                "Antihistamines (for allergies)",
                                "Saline nasal rinses",
                                "Humidifiers",
                                "Treating underlying cause (e.g., cold, allergies)"
                            ]
                        },
                        {
                            "name": "Sore Throat",
                            "description": "Pain, irritation, or scratchiness in the throat, often made worse by swallowing.",
                            "medical_practice": "ENT, General Practice",
                            "possible_treatments": [
                                "Pain relievers (e.g., acetaminophen, ibuprofen)",
                                "Throat lozenges/sprays",
                                "Warm salt water gargles",
                                "Hydration",
                                "Treating underlying cause"
                            ]
                        },
                        {
                            "name": "Dizziness/Vertigo",
                            "description": "Feeling lightheaded, unsteady, or experiencing a sensation that the room is spinning.",
                            "medical_practice": "Neurology, ENT, General Practice",
                            "possible_treatments": [
                                "Vestibular rehabilitation",
                                "Medications to reduce nausea or dizziness",
                                "Hydration",
                                "Treating underlying cause"
                            ]
                        }
                    ]
                }
            ]
        
        # Parse data into structures and documents
        parsed_symptoms = []
        symptom_descriptions = []
        
        for category in symptoms_data:
            parsed_category = Symptoms(category["category"], [])
            for symptom_data in category["symptoms"]:
                symptom = Symptom(
                    name=symptom_data["name"],
                    description=symptom_data["description"],
                    medical_practice=symptom_data["medical_practice"],
                    possible_treatments=symptom_data["possible_treatments"]
                )
                parsed_category.symptoms.append(symptom)
                parsed_symptoms.append(symptom)
                
                # Create document for vector store
                symptom_descriptions.append(
                    Document(
                        page_content=symptom.description.strip(),
                        metadata=asdict(symptom)
                    )
                )
        
        self.parsed_symptoms = parsed_symptoms
        click.echo(f"📄 Loaded {len(symptom_descriptions)} symptom documents")
        return symptom_descriptions
    
    async def ingest_data(self, documents: List[Document]):
        """
        Ingest documents into both vector store and graph store
        
        Args:
            documents: List of Document objects to ingest
        """
        try:
            # Add documents to vector store (this calculates embeddings automatically)
            await self.vector_store.aadd_documents(documents)
            click.echo("✅ Documents added to vector store")
            
            # Create graph documents for graph store
            graph_documents = []
            
            for idx, doc in enumerate(documents):
                symptom = self.parsed_symptoms[idx]
                
                # Create nodes
                treatment_nodes = {}
                symptom_node = Node(
                    id=symptom.name, 
                    type="Symptom", 
                    properties=asdict(symptom)
                )
                
                for treatment in symptom.possible_treatments:
                    treatment_nodes[treatment] = Node(
                        id=treatment, 
                        type="Treatment", 
                        properties={"name": treatment}
                    )
                
                # Create medical practice node
                practice_node = Node(
                    id=symptom.medical_practice,
                    type="Practice",
                    properties={"name": symptom.medical_practice}
                )
                
                nodes = list(treatment_nodes.values())
                nodes.extend([symptom_node, practice_node])
                
                # Create relationships
                relationships = []
                
                # Treatment -> Treats -> Symptom relationships
                for treatment in symptom.possible_treatments:
                    relationships.append(
                        Relationship(
                            source=treatment_nodes[treatment],
                            target=symptom_node,
                            type="Treats"
                        )
                    )
                
                # Practice -> Attends -> Symptom relationship
                relationships.append(
                    Relationship(
                        source=practice_node,
                        target=symptom_node,
                        type="Attends"
                    )
                )
                
                graph_documents.append(
                    GraphDocument(
                        nodes=nodes, 
                        relationships=relationships, 
                        source=doc
                    )
                )
            
            # Store graph documents
            await self.graph_store.aadd_graph_documents(graph_documents, include_source=True)
            click.echo("✅ Graph documents added to graph store")
            
        except Exception as e:
            click.echo(f"❌ Error during data ingestion: {e}")
            raise
    
    async def vector_search(self, query: str, k: int = 3) -> List[Document]:
        """
        Perform vector similarity search
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        try:
            docs = await self.vector_store.asimilarity_search(query, k=k)
            return docs
        except Exception as e:
            click.echo(f"❌ Error during vector search: {e}")
            return []
    
    def get_document_names(self, docs: List[Document]) -> List[str]:
        """Extract symptom names from retrieved documents"""
        names = []
        for doc in docs:
            if "name" in doc.metadata:
                names.append(doc.metadata["name"])
        return names
    
    async def query_graph(self, question: str) -> str:
        """
        Query the knowledge graph using the QA chain
        
        Args:
            question: Question to ask the graph
            
        Returns:
            Generated response from the graph QA chain
        """
        try:
            # Create the graph QA chain
            chain = SurrealDBGraphQAChain.from_llm(
                self.chat_model,
                graph=self.graph_store,
                verbose=self.verbose
            )
            
            # Run the chain
            response = await chain.arun(question)
            return response
            
        except Exception as e:
            click.echo(f"❌ Error during graph query: {e}")
            return f"Error processing question: {e}"
    
    async def chat_session(self):
        """
        Interactive chat session with the GraphRAG system
        """
        click.echo("\n🤖 GraphRAG Chatbot is ready!")
        click.echo("Ask me about symptoms, treatments, or medical practices.")
        click.echo("Type 'quit' to exit.\n")
        
        while True:
            try:
                # Get user input
                query = click.prompt(
                    click.style("What are your symptoms or question?", fg="green"),
                    type=str
                )
                
                if query.lower() in ['quit', 'exit', 'bye']:
                    click.echo("👋 Goodbye!")
                    break
                
                click.echo("\n🔍 Searching for relevant information...")
                
                # Step 1: Find relevant documents using vector search
                docs = await self.vector_search(query, k=3)
                symptoms = self.get_document_names(docs)
                
                if docs:
                    click.echo(f"📋 Found relevant symptoms: {', '.join(symptoms)}")
                    
                    # Display vector search results
                    click.echo("\n📄 Vector Search Results:")
                    for i, doc in enumerate(docs, 1):
                        click.echo(f"{i}. {doc.page_content}")
                
                # Step 2: Query the graph for medical practices
                if symptoms:
                    click.echo("\n🏥 Querying for medical practices...")
                    practices_question = f"what medical practices can help with {', '.join(symptoms)}"
                    practices_response = await self.query_graph(practices_question)
                    click.echo(f"\n🏥 Medical Practices:\n{practices_response}")
                    
                    # Step 3: Query the graph for treatments
                    click.echo("\n💊 Querying for treatments...")
                    treatments_question = f"what treatments can help with {', '.join(symptoms)}"
                    treatments_response = await self.query_graph(treatments_question)
                    click.echo(f"\n💊 Treatments:\n{treatments_response}")
                else:
                    click.echo("❌ No relevant symptoms found. Please try rephrasing your question.")
                
                click.echo("\n" + "="*50 + "\n")
                
            except KeyboardInterrupt:
                click.echo("\n👋 Goodbye!")
                break
            except Exception as e:
                click.echo(f"❌ An error occurred: {e}")
    
    async def close(self):
        """Clean up connections"""
        if self.conn:
            await self.conn.close()


async def main():
    """
    Main function to demonstrate GraphRAG chatbot functionality
    """
    click.echo("🚀 Initializing GraphRAG Chatbot with SurrealDB...")
    
    # Configuration - adjust these based on your setup
    config = {
        "surrealdb_url": os.getenv("SURREALDB_URL", "ws://localhost:8000/rpc"),
        "namespace": os.getenv("SURREALDB_NS", "test"),
        "database": os.getenv("SURREALDB_DB", "test"), 
        "username": os.getenv("SURREALDB_USER", "root"),
        "password": os.getenv("SURREALDB_PASS", "root"),
        "use_openai": os.getenv("USE_OPENAI", "false").lower() == "true",
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "model_name": os.getenv("MODEL_NAME", "llama3.2"),
        "verbose": os.getenv("VERBOSE", "false").lower() == "true"
    }
    
    # Initialize chatbot
    chatbot = GraphRAGChatbot(**config)
    
    try:
        # Initialize connection
        await chatbot.initialize_connection()
        
        # Load and ingest sample data
        click.echo("📊 Loading sample medical data...")
        documents = chatbot.load_sample_data()
        
        click.echo("📥 Ingesting data into vector and graph stores...")
        await chatbot.ingest_data(documents)
        
        # Start interactive chat session
        await chatbot.chat_session()
        
    except Exception as e:
        click.echo(f"❌ Fatal error: {e}")
    finally:
        await chatbot.close()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())