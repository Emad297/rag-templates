"""
Knowledge Graph Building module for GraphRAG with SurrealDB
"""

import asyncio
from typing import Dict, List, Any, Set, Tuple, Optional
from dataclasses import dataclass
import json

from rich.console import Console
from rich.table import Table
from rich.progress import Progress

console = Console()


@dataclass
class GraphNode:
    """Represents a node in the knowledge graph"""
    id: str
    type: str
    properties: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "properties": self.properties
        }


@dataclass
class GraphEdge:
    """Represents an edge/relationship in the knowledge graph"""
    source: str
    target: str
    relationship_type: str
    properties: Dict[str, Any]
    weight: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relationship_type": self.relationship_type,
            "properties": self.properties,
            "weight": self.weight
        }


class KnowledgeGraphBuilder:
    """Advanced knowledge graph builder for medical domain"""
    
    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.node_types = set()
        self.relationship_types = set()
    
    def add_node(self, node_id: str, node_type: str, properties: Dict[str, Any]) -> None:
        """Add a node to the knowledge graph"""
        self.nodes[node_id] = GraphNode(node_id, node_type, properties)
        self.node_types.add(node_type)
    
    def add_edge(self, source: str, target: str, relationship_type: str, 
                 properties: Dict[str, Any] = None, weight: float = 1.0) -> None:
        """Add an edge to the knowledge graph"""
        if properties is None:
            properties = {}
        
        edge = GraphEdge(source, target, relationship_type, properties, weight)
        self.edges.append(edge)
        self.relationship_types.add(relationship_type)
    
    def build_from_symptoms(self, symptoms_data: List[Any]) -> None:
        """Build knowledge graph from symptoms data"""
        with Progress() as progress:
            task = progress.add_task("Building knowledge graph...", total=len(symptoms_data))
            
            for symptom in symptoms_data:
                # Add symptom node
                symptom_id = f"symptom_{symptom.name.lower().replace(' ', '_')}"
                self.add_node(symptom_id, "symptom", {
                    "name": symptom.name,
                    "description": symptom.description,
                    "medical_practice": symptom.medical_practice
                })
                
                # Add treatment nodes and relationships
                for treatment in symptom.possible_treatments:
                    treatment_id = f"treatment_{treatment.lower().replace(' ', '_')}"
                    
                    # Add treatment node if not exists
                    if treatment_id not in self.nodes:
                        self.add_node(treatment_id, "treatment", {
                            "name": treatment,
                            "type": "treatment"
                        })
                    
                    # Add treats relationship
                    self.add_edge(treatment_id, symptom_id, "treats", {
                        "effectiveness": "unknown",
                        "confidence": 0.8
                    })
                
                # Add medical practice node and relationship
                practice_id = f"practice_{symptom.medical_practice.lower().replace(' ', '_')}"
                
                if practice_id not in self.nodes:
                    self.add_node(practice_id, "medical_practice", {
                        "name": symptom.medical_practice,
                        "specialization": symptom.medical_practice
                    })
                
                # Add practiced_by relationship
                self.add_edge(symptom_id, practice_id, "practiced_by", {
                    "primary": True
                })
                
                progress.update(task, advance=1)
        
        console.print(f"✅ Built knowledge graph with {len(self.nodes)} nodes and {len(self.edges)} edges", style="green")
    
    def find_shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """Find shortest path between two nodes using BFS"""
        if source not in self.nodes or target not in self.nodes:
            return None
        
        if source == target:
            return [source]
        
        # Build adjacency list
        adjacency = {}
        for node_id in self.nodes:
            adjacency[node_id] = []
        
        for edge in self.edges:
            adjacency[edge.source].append(edge.target)
            adjacency[edge.target].append(edge.source)  # Undirected graph
        
        # BFS
        from collections import deque
        
        queue = deque([(source, [source])])
        visited = {source}
        
        while queue:
            current, path = queue.popleft()
            
            for neighbor in adjacency[current]:
                if neighbor == target:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def get_neighbors(self, node_id: str, relationship_type: str = None) -> List[Tuple[str, str, float]]:
        """Get neighboring nodes with relationship types and weights"""
        neighbors = []
        
        for edge in self.edges:
            if edge.source == node_id:
                if relationship_type is None or edge.relationship_type == relationship_type:
                    neighbors.append((edge.target, edge.relationship_type, edge.weight))
            elif edge.target == node_id:
                if relationship_type is None or edge.relationship_type == relationship_type:
                    neighbors.append((edge.source, edge.relationship_type, edge.weight))
        
        return neighbors
    
    def calculate_node_centrality(self) -> Dict[str, float]:
        """Calculate degree centrality for all nodes"""
        centrality = {}
        
        for node_id in self.nodes:
            degree = len(self.get_neighbors(node_id))
            centrality[node_id] = degree / max(1, len(self.nodes) - 1)
        
        return centrality
    
    def get_subgraph(self, node_id: str, depth: int = 2) -> 'KnowledgeGraphBuilder':
        """Extract subgraph around a specific node"""
        subgraph = KnowledgeGraphBuilder()
        
        # BFS to find nodes within depth
        from collections import deque
        
        queue = deque([(node_id, 0)])
        visited = {node_id}
        nodes_to_include = {node_id}
        
        while queue:
            current, current_depth = queue.popleft()
            
            if current_depth < depth:
                neighbors = self.get_neighbors(current)
                for neighbor_id, _, _ in neighbors:
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        nodes_to_include.add(neighbor_id)
                        queue.append((neighbor_id, current_depth + 1))
        
        # Add nodes to subgraph
        for node_id in nodes_to_include:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                subgraph.add_node(node.id, node.type, node.properties)
        
        # Add edges to subgraph
        for edge in self.edges:
            if edge.source in nodes_to_include and edge.target in nodes_to_include:
                subgraph.add_edge(edge.source, edge.target, edge.relationship_type, 
                                edge.properties, edge.weight)
        
        return subgraph
    
    def export_to_networkx(self):
        """Export to NetworkX graph for advanced analysis"""
        try:
            import networkx as nx
            
            G = nx.Graph()
            
            # Add nodes
            for node in self.nodes.values():
                G.add_node(node.id, **node.properties, node_type=node.type)
            
            # Add edges
            for edge in self.edges:
                G.add_edge(edge.source, edge.target, 
                          relationship_type=edge.relationship_type,
                          weight=edge.weight, **edge.properties)
            
            return G
            
        except ImportError:
            console.print("⚠️ NetworkX not installed", style="yellow")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics"""
        stats = {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "node_types": list(self.node_types),
            "relationship_types": list(self.relationship_types),
            "nodes_by_type": {},
            "edges_by_type": {},
            "average_degree": 0
        }
        
        # Count nodes by type
        for node in self.nodes.values():
            node_type = node.type
            stats["nodes_by_type"][node_type] = stats["nodes_by_type"].get(node_type, 0) + 1
        
        # Count edges by type
        for edge in self.edges:
            rel_type = edge.relationship_type
            stats["edges_by_type"][rel_type] = stats["edges_by_type"].get(rel_type, 0) + 1
        
        # Calculate average degree
        if self.nodes:
            total_degree = sum(len(self.get_neighbors(node_id)) for node_id in self.nodes)
            stats["average_degree"] = total_degree / len(self.nodes)
        
        return stats
    
    def display_statistics(self) -> None:
        """Display graph statistics in a formatted table"""
        stats = self.get_statistics()
        
        # Main statistics table
        main_table = Table(title="Knowledge Graph Statistics")
        main_table.add_column("Metric", style="cyan")
        main_table.add_column("Value", style="green")
        
        main_table.add_row("Total Nodes", str(stats["total_nodes"]))
        main_table.add_row("Total Edges", str(stats["total_edges"]))
        main_table.add_row("Node Types", str(len(stats["node_types"])))
        main_table.add_row("Relationship Types", str(len(stats["relationship_types"])))
        main_table.add_row("Average Degree", f"{stats['average_degree']:.2f}")
        
        console.print(main_table)
        
        # Node types breakdown
        if stats["nodes_by_type"]:
            node_table = Table(title="Nodes by Type")
            node_table.add_column("Node Type", style="blue")
            node_table.add_column("Count", style="green")
            
            for node_type, count in stats["nodes_by_type"].items():
                node_table.add_row(node_type, str(count))
            
            console.print(node_table)
        
        # Relationship types breakdown
        if stats["edges_by_type"]:
            edge_table = Table(title="Relationships by Type")
            edge_table.add_column("Relationship Type", style="magenta")
            edge_table.add_column("Count", style="green")
            
            for rel_type, count in stats["edges_by_type"].items():
                edge_table.add_row(rel_type, str(count))
            
            console.print(edge_table)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export entire graph to dictionary"""
        return {
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "edges": [edge.to_dict() for edge in self.edges],
            "statistics": self.get_statistics()
        }
    
    def save_to_file(self, file_path: str) -> bool:
        """Save graph to JSON file"""
        try:
            graph_data = self.to_dict()
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(graph_data, file, indent=2, ensure_ascii=False)
            
            console.print(f"✅ Graph saved to {file_path}", style="green")
            return True
            
        except Exception as e:
            console.print(f"❌ Error saving graph: {str(e)}", style="red")
            return False