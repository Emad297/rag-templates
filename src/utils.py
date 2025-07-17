"""
Utility functions for GraphRAG with SurrealDB tutorial
"""

import asyncio
import yaml
import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.progress import Progress

console = Console()


@dataclass
class Symptom:
    """Data model for medical symptoms"""
    name: str
    description: str
    medical_practice: str
    possible_treatments: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_document_content(self) -> str:
        """Convert symptom to document content for embeddings"""
        return f"Symptom: {self.name}\nDescription: {self.description}\nMedical Practice: {self.medical_practice}\nTreatments: {', '.join(self.possible_treatments)}"


def load_symptoms_from_yaml(file_path: str) -> List[Symptom]:
    """
    Load symptoms data from YAML file
    
    Args:
        file_path: Path to the YAML file
        
    Returns:
        List of Symptom objects
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        
        symptoms = []
        for category in data:
            for symptom_data in category['symptoms']:
                symptom = Symptom(
                    name=symptom_data['name'],
                    description=symptom_data['description'],
                    medical_practice=symptom_data['medical_practice'],
                    possible_treatments=symptom_data['possible_treatments']
                )
                symptoms.append(symptom)
        
        console.print(f"✅ Loaded {len(symptoms)} symptoms", style="green")
        return symptoms
        
    except Exception as e:
        console.print(f"❌ Error loading symptoms: {str(e)}", style="red")
        return []


def save_to_json(data: Any, file_path: str) -> bool:
    """Save data to JSON file"""
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
        console.print(f"✅ Saved data to {file_path}", style="green")
        return True
    except Exception as e:
        console.print(f"❌ Error saving to JSON: {str(e)}", style="red")
        return False


def load_from_json(file_path: str) -> Optional[Any]:
    """Load data from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        console.print(f"✅ Loaded data from {file_path}", style="green")
        return data
    except Exception as e:
        console.print(f"❌ Error loading from JSON: {str(e)}", style="red")
        return None


def display_symptoms_table(symptoms: List[Symptom], max_rows: int = 10) -> None:
    """Display symptoms in a formatted table"""
    table = Table(title=f"Medical Symptoms ({len(symptoms)} total)")
    table.add_column("Name", style="cyan")
    table.add_column("Medical Practice", style="green")
    table.add_column("Treatments", style="yellow")
    
    for symptom in symptoms[:max_rows]:
        treatments = ", ".join(symptom.possible_treatments[:2])
        if len(symptom.possible_treatments) > 2:
            treatments += "..."
        
        table.add_row(
            symptom.name,
            symptom.medical_practice,
            treatments
        )
    
    if len(symptoms) > max_rows:
        table.add_row("...", "...", f"... and {len(symptoms) - max_rows} more")
    
    console.print(table)


def create_config_from_env() -> Dict[str, str]:
    """Create configuration from environment variables"""
    return {
        "SURREALDB_URL": os.getenv("SURREALDB_URL", "ws://localhost:8000/rpc"),
        "SURREALDB_USERNAME": os.getenv("SURREALDB_USERNAME", "root"),
        "SURREALDB_PASSWORD": os.getenv("SURREALDB_PASSWORD", "root"),
        "SURREALDB_NAMESPACE": os.getenv("SURREALDB_NAMESPACE", "medical"),
        "SURREALDB_DATABASE": os.getenv("SURREALDB_DATABASE", "graphrag"),
        "OLLAMA_BASE_URL": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "OLLAMA_MODEL": os.getenv("OLLAMA_MODEL", "llama3.2"),
        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "llama3.2"),
    }


async def test_surrealdb_connection(url: str, username: str, password: str) -> bool:
    """Test SurrealDB connection"""
    try:
        from surrealdb import Surreal
        
        db = Surreal()
        await db.connect(url)
        await db.signin({"user": username, "pass": password})
        await db.close()
        
        console.print("✅ SurrealDB connection successful", style="green")
        return True
        
    except Exception as e:
        console.print(f"❌ SurrealDB connection failed: {str(e)}", style="red")
        return False


def calculate_similarity_matrix(embeddings: List[List[float]]) -> List[List[float]]:
    """Calculate cosine similarity matrix for embeddings"""
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    if not embeddings:
        return []
    
    embeddings_array = np.array(embeddings)
    similarity_matrix = cosine_similarity(embeddings_array)
    
    return similarity_matrix.tolist()


def find_most_similar(query_embedding: List[float], embeddings: List[List[float]], k: int = 5) -> List[int]:
    """Find k most similar embeddings to query"""
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    if not embeddings or not query_embedding:
        return []
    
    query_array = np.array(query_embedding).reshape(1, -1)
    embeddings_array = np.array(embeddings)
    
    similarities = cosine_similarity(query_array, embeddings_array)[0]
    top_indices = np.argsort(similarities)[::-1][:k]
    
    return top_indices.tolist()


class ProgressTracker:
    """Helper class for tracking progress in long operations"""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.description = description
        self.progress = None
        self.task = None
    
    def __enter__(self):
        self.progress = Progress()
        self.task = self.progress.add_task(self.description, total=self.total)
        self.progress.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.progress:
            self.progress.stop()
    
    def update(self, advance: int = 1):
        if self.progress and self.task:
            self.progress.update(self.task, advance=advance)