from docx import Document
import numpy as np
from typing import Any, Dict, List
import subprocess
import os


def normalize_vector(vector: List[float]) -> List[float]:
    """Normalize vector for cosine similarity"""
    vector_array = np.array(vector)
    norm = np.linalg.norm(vector_array)
    if norm == 0:
        return vector  # Return as-is if zero vector
    normalized = vector_array / norm
    return normalized.tolist()

def validate_file_type(filename: str) -> None:
    """
    Validate if file type is supported. Raises ValueError for unsupported types.
    """
    SUPPORTED_EXTENSIONS = {'.docx', '.pdf', '.txt'}
    
    file_extension = os.path.splitext(filename.lower())[1]
    
    if not file_extension:
        raise ValueError(f"File '{filename}' has no extension. Supported types: {', '.join(SUPPORTED_EXTENSIONS)}")
    
    if file_extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{file_extension}' for file '{filename}'. "
            f"Supported types: {', '.join(SUPPORTED_EXTENSIONS)}. "
            f"Please convert to a supported format."
        )

def extract_text_from_file(file_path: str) -> str:
    """Extract text from various file types including tables in Word documents"""
    ext = file_path.lower()
    
    if ext.endswith(".pdf"):
        from pdfminer.high_level import extract_text
        return extract_text(file_path)
    elif ext.endswith(".docx"):
        from docx import Document
        doc = Document(file_path)
        
        # Extract text from paragraphs
        text_parts = [p.text for p in doc.paragraphs if p.text.strip()]
        
        # Extract text from tables
        for table in doc.tables:
            table_text = extract_table_text(table)
            if table_text:
                text_parts.append(table_text)
        
        return "\n".join(text_parts)
    elif ext.endswith(".txt"):
        with open(file_path, "r", encoding='utf-8') as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

def extract_table_text(table) -> str:
    """Extract text from a Word table and format it for LLM understanding"""
    table_lines = []
    
    for row_idx, row in enumerate(table.rows):
        row_cells = []
        for cell in row.cells:
            cell_text = cell.text.strip()
            if cell_text:
                row_cells.append(cell_text)
        
        if row_cells:
            # Format as pipe-separated table for LLM
            table_lines.append(" | ".join(row_cells))
    
    if table_lines:
        return "TABLE:\n" + "\n".join(table_lines)
    return ""


# Optional: Helper function for tag-based searching
def filter_templates_by_tags(templates: List[Dict[str, Any]], required_tags: List[str] = None, 
                           any_tags: List[str] = None, exclude_tags: List[str] = None) -> List[Dict[str, Any]]:
    """
    Filter templates based on tag criteria
    """
    if not any([required_tags, any_tags, exclude_tags]):
        return templates
    
    filtered_templates = []
    
    for template in templates:
        template_tags = template.get("tags", [])
        
        # Check required tags (ALL must be present)
        if required_tags and not all(tag in template_tags for tag in required_tags):
            continue
        
        # Check any tags (AT LEAST ONE must be present)
        if any_tags and not any(tag in template_tags for tag in any_tags):
            continue
        
        # Check exclude tags (NONE can be present)
        if exclude_tags and any(tag in template_tags for tag in exclude_tags):
            continue
        
        filtered_templates.append(template)
    
    return filtered_templates
    
