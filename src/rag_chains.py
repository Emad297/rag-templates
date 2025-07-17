"""
RAG Chains implementation for GraphRAG with SurrealDB
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

from rich.console import Console

console = Console()


@dataclass
class RetrievalResult:
    """Result from retrieval operation"""
    content: str
    metadata: Dict[str, Any]
    score: float
    source_type: str  # 'vector', 'graph', 'hybrid'


@dataclass
class QueryContext:
    """Context information for query processing"""
    original_query: str
    processed_query: str
    intent: str
    entities: List[str]
    relationships: List[str]


class BaseRAGChain(ABC):
    """Abstract base class for RAG chains"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    async def retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """Retrieve relevant context for the query"""
        pass
    
    @abstractmethod
    def generate_response(self, query: str, context: List[RetrievalResult]) -> str:
        """Generate response based on retrieved context"""
        pass
    
    async def query(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Main query interface"""
        # Retrieve context
        context = await self.retrieve(query, k)
        
        # Generate response
        response = self.generate_response(query, context)
        
        return {
            "query": query,
            "response": response,
            "context": context,
            "metadata": {
                "num_results": len(context),
                "sources": [r.source_type for r in context]
            }
        }


class VectorRAGChain(BaseRAGChain):
    """Traditional vector-based RAG chain"""
    
    def __init__(self, config: Dict[str, Any], vector_store: Dict[str, Any], embedding_manager: Any):
        super().__init__(config)
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
    
    async def retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """Retrieve using vector similarity only"""
        vector_results = self.embedding_manager.similarity_search(query, self.vector_store, k)
        
        retrieval_results = []
        for result in vector_results:
            retrieval_result = RetrievalResult(
                content=result['content'],
                metadata=result['metadata'],
                score=result['score'],
                source_type='vector'
            )
            retrieval_results.append(retrieval_result)
        
        return retrieval_results
    
    def generate_response(self, query: str, context: List[RetrievalResult]) -> str:
        """Generate response using vector context only"""
        if not context:
            return "I couldn't find any relevant information for your query."
        
        # Simple response generation
        primary_result = context[0]
        response_parts = [
            f"Based on your query '{query}', here's what I found:",
            f"Primary match: {primary_result.metadata.get('name', 'Unknown')}",
            f"Treatments: {', '.join(primary_result.metadata.get('treatments', []))}"
        ]
        
        if len(context) > 1:
            other_matches = [r.metadata.get('name', 'Unknown') for r in context[1:3]]
            response_parts.append(f"Also relevant: {', '.join(other_matches)}")
        
        response_parts.append("\n⚠️ This is for educational purposes only. Consult a healthcare professional.")
        
        return " ".join(response_parts)


class GraphRAGChain(BaseRAGChain):
    """Advanced GraphRAG chain combining vector and graph retrieval"""
    
    def __init__(self, config: Dict[str, Any], vector_store: Dict[str, Any], 
                 embedding_manager: Any, knowledge_graph: Any):
        super().__init__(config)
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.knowledge_graph = knowledge_graph
    
    def extract_entities(self, query: str) -> List[str]:
        """Extract potential entities from query"""
        # Simple keyword extraction for demo
        medical_keywords = [
            'headache', 'fever', 'nausea', 'pain', 'cough', 'fatigue',
            'dizziness', 'rash', 'breathing', 'chest', 'stomach', 'skin'
        ]
        
        query_lower = query.lower()
        entities = [keyword for keyword in medical_keywords if keyword in query_lower]
        return entities
    
    def expand_with_graph_context(self, vector_results: List[Dict], depth: int = 2) -> Dict[str, Any]:
        """Expand vector results with graph context"""
        graph_context = {}
        
        for result in vector_results:
            symptom_name = result['metadata']['name']
            
            # Get direct relationships
            neighbors = self.knowledge_graph.get_neighbors(
                f"symptom_{symptom_name.lower().replace(' ', '_')}"
            )
            
            graph_context[symptom_name] = {
                "direct_relationships": [],
                "connected_treatments": [],
                "medical_practices": []
            }
            
            for neighbor_id, rel_type, weight in neighbors:
                neighbor_node = self.knowledge_graph.nodes.get(neighbor_id)
                if neighbor_node:
                    relationship_info = {
                        "entity": neighbor_node.properties.get('name', neighbor_id),
                        "type": neighbor_node.type,
                        "relationship": rel_type,
                        "weight": weight
                    }
                    
                    graph_context[symptom_name]["direct_relationships"].append(relationship_info)
                    
                    if neighbor_node.type == "treatment":
                        graph_context[symptom_name]["connected_treatments"].append(
                            neighbor_node.properties.get('name', neighbor_id)
                        )
                    elif neighbor_node.type == "medical_practice":
                        graph_context[symptom_name]["medical_practices"].append(
                            neighbor_node.properties.get('name', neighbor_id)
                        )
        
        return graph_context
    
    async def retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """Enhanced retrieval combining vector and graph"""
        # Step 1: Vector similarity search
        vector_results = self.embedding_manager.similarity_search(query, self.vector_store, k)
        
        # Step 2: Expand with graph context
        graph_context = self.expand_with_graph_context(vector_results)
        
        # Step 3: Create enhanced retrieval results
        retrieval_results = []
        
        for result in vector_results:
            # Original vector result
            vector_retrieval = RetrievalResult(
                content=result['content'],
                metadata=result['metadata'],
                score=result['score'],
                source_type='vector'
            )
            retrieval_results.append(vector_retrieval)
            
            # Add graph-enhanced context
            symptom_name = result['metadata']['name']
            if symptom_name in graph_context:
                graph_info = graph_context[symptom_name]
                
                # Create enhanced content with graph information
                enhanced_content = result['content']
                if graph_info['connected_treatments']:
                    enhanced_content += f"\nConnected treatments: {', '.join(graph_info['connected_treatments'])}"
                if graph_info['medical_practices']:
                    enhanced_content += f"\nMedical specialties: {', '.join(graph_info['medical_practices'])}"
                
                graph_retrieval = RetrievalResult(
                    content=enhanced_content,
                    metadata={**result['metadata'], 'graph_context': graph_info},
                    score=result['score'] * 1.2,  # Boost score for graph-enhanced results
                    source_type='hybrid'
                )
                retrieval_results.append(graph_retrieval)
        
        # Sort by score and return top k
        retrieval_results.sort(key=lambda x: x.score, reverse=True)
        return retrieval_results[:k]
    
    def generate_response(self, query: str, context: List[RetrievalResult]) -> str:
        """Generate enhanced response using both vector and graph context"""
        if not context:
            return "I couldn't find any relevant information for your query."
        
        # Separate vector and hybrid results
        vector_results = [r for r in context if r.source_type == 'vector']
        hybrid_results = [r for r in context if r.source_type == 'hybrid']
        
        response_parts = []
        response_parts.append(f"Based on your query '{query}', here's a comprehensive analysis:")
        
        # Primary recommendation
        if hybrid_results:
            primary = hybrid_results[0]
            response_parts.append(f"\n🎯 Primary recommendation: {primary.metadata.get('name', 'Unknown')}")
            
            # Treatment information
            treatments = primary.metadata.get('treatments', [])
            if treatments:
                response_parts.append(f"💊 Recommended treatments: {', '.join(treatments)}")
            
            # Graph context
            if 'graph_context' in primary.metadata:
                graph_info = primary.metadata['graph_context']
                
                if graph_info.get('medical_practices'):
                    response_parts.append(f"🏥 Relevant specialties: {', '.join(graph_info['medical_practices'])}")
                
                # Related conditions
                related_treatments = graph_info.get('connected_treatments', [])
                if related_treatments and len(related_treatments) > len(treatments):
                    additional = [t for t in related_treatments if t not in treatments]
                    if additional:
                        response_parts.append(f"🔗 Additional treatment options: {', '.join(additional[:3])}")
        
        # Additional relevant information
        if len(context) > 1:
            other_symptoms = []
            for result in context[1:4]:  # Show up to 3 additional results
                name = result.metadata.get('name', 'Unknown')
                if name not in [r.metadata.get('name') for r in hybrid_results]:
                    other_symptoms.append(name)
            
            if other_symptoms:
                response_parts.append(f"\n📋 Also consider: {', '.join(other_symptoms)}")
        
        # Confidence and sources
        avg_score = sum(r.score for r in context) / len(context)
        confidence = "High" if avg_score > 2 else "Medium" if avg_score > 1 else "Low"
        response_parts.append(f"\n📊 Confidence: {confidence} (based on {len(context)} sources)")
        
        # Safety disclaimer
        response_parts.append("\n⚠️ This information is for educational purposes only. Always consult a qualified healthcare professional for medical advice, diagnosis, or treatment.")
        
        return "".join(response_parts)


class HybridRAGChain(BaseRAGChain):
    """Hybrid RAG chain that combines multiple retrieval strategies"""
    
    def __init__(self, config: Dict[str, Any], vector_chain: VectorRAGChain, graph_chain: GraphRAGChain):
        super().__init__(config)
        self.vector_chain = vector_chain
        self.graph_chain = graph_chain
    
    async def retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """Retrieve using multiple strategies and merge results"""
        # Get results from both chains
        vector_results = await self.vector_chain.retrieve(query, k)
        graph_results = await self.graph_chain.retrieve(query, k)
        
        # Merge and deduplicate
        all_results = vector_results + graph_results
        
        # Remove duplicates based on content similarity
        unique_results = []
        seen_content = set()
        
        for result in all_results:
            content_key = result.metadata.get('name', result.content[:50])
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_results.append(result)
        
        # Sort by score and return top k
        unique_results.sort(key=lambda x: x.score, reverse=True)
        return unique_results[:k]
    
    def generate_response(self, query: str, context: List[RetrievalResult]) -> str:
        """Generate response using the most appropriate context"""
        # Prefer hybrid results, fall back to graph, then vector
        hybrid_context = [r for r in context if r.source_type == 'hybrid']
        if hybrid_context:
            return self.graph_chain.generate_response(query, hybrid_context)
        
        graph_context = [r for r in context if r.source_type == 'graph']
        if graph_context:
            return self.graph_chain.generate_response(query, graph_context)
        
        # Fall back to vector chain
        return self.vector_chain.generate_response(query, context)


class QueryAnalyzer:
    """Analyzes queries to determine optimal retrieval strategy"""
    
    def __init__(self):
        self.medical_entities = {
            'symptoms': ['headache', 'fever', 'nausea', 'pain', 'cough', 'fatigue', 'dizziness'],
            'treatments': ['medication', 'therapy', 'surgery', 'rest', 'exercise'],
            'practices': ['cardiology', 'neurology', 'dermatology', 'general', 'emergency']
        }
    
    def analyze_query(self, query: str) -> QueryContext:
        """Analyze query to extract intent and entities"""
        query_lower = query.lower()
        
        # Extract entities
        entities = []
        for entity_type, entity_list in self.medical_entities.items():
            for entity in entity_list:
                if entity in query_lower:
                    entities.append(entity)
        
        # Determine intent
        intent = "symptom_lookup"  # Default
        if any(word in query_lower for word in ['treat', 'treatment', 'help', 'cure']):
            intent = "treatment_seeking"
        elif any(word in query_lower for word in ['relate', 'connect', 'similar']):
            intent = "relationship_query"
        elif any(word in query_lower for word in ['what is', 'define', 'explain']):
            intent = "definition_query"
        
        # Extract potential relationships
        relationships = []
        if 'cause' in query_lower:
            relationships.append('causes')
        if 'treat' in query_lower:
            relationships.append('treats')
        if 'relate' in query_lower:
            relationships.append('related_to')
        
        return QueryContext(
            original_query=query,
            processed_query=query_lower,
            intent=intent,
            entities=entities,
            relationships=relationships
        )
    
    def recommend_chain(self, context: QueryContext) -> str:
        """Recommend which RAG chain to use based on query analysis"""
        # If query involves relationships or multiple entities, use GraphRAG
        if (len(context.entities) > 1 or 
            context.relationships or 
            context.intent in ['relationship_query', 'treatment_seeking']):
            return 'graph'
        
        # For simple symptom lookups, vector search is sufficient
        if context.intent == 'symptom_lookup' and len(context.entities) == 1:
            return 'vector'
        
        # Default to hybrid for complex queries
        return 'hybrid'