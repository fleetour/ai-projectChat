import re
from typing import List, Dict

def count_placeholders(text: str) -> Dict[str, int]:
    """
    Count different types of placeholders in the template
    """
    placeholder_patterns = {
        "underscores": r'_{3,}',           # ___, ____
        "brackets": r'\[.*?\]',            # [placeholder]
        "parentheses": r'\(.*?\)',         # (placeholder)
        "angle_brackets": r'<.*?>',        # <placeholder>
        "curly_braces": r'\{.*?\}',        # {placeholder}
        "todo_indicators": r'\b(TODO|FILL|INSERT|ADD|COMPLETE)\b',
        "question_marks": r'\?{2,}',       # ??, ???
        "empty_lines": r'^\s*$',           # Blank lines between sections
        "short_paragraphs": r'^.{1,50}$',  # Very short lines that need expansion
    }
    
    counts = {}
    for pattern_name, pattern in placeholder_patterns.items():
        matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
        counts[pattern_name] = len(matches)
    
    # Also count generic gaps (sentences that seem incomplete)
    incomplete_sentences = count_incomplete_sentences(text)
    counts["incomplete_sentences"] = incomplete_sentences
    
    # Total placeholders
    counts["total"] = sum(counts.values())
    
    return counts

def count_incomplete_sentences(text: str) -> int:
    """
    Count sentences that seem incomplete or need continuation
    """
    # Split into sentences (simple approach)
    sentences = re.split(r'[.!?]+', text)
    
    incomplete_count = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Heuristics for incomplete sentences
        if (len(sentence) < 20 or  # Very short
            sentence.endswith((':', '-', '—', '--')) or  # Ends with continuation markers
            any(word in sentence.lower() for word in ['such as', 'including', 'for example', 'e.g.']) or
            re.search(r'\b(should|will|can|must|need to)\b', sentence.lower()) and len(sentence) < 50):
            incomplete_count += 1
    
    return incomplete_count

def extract_key_topics(text: str, max_topics: int = 10) -> List[str]:
    """
    Extract key topics and themes from the template
    """
    # Common document sections and topics
    common_sections = {
        'project': ['project', 'initiative', 'program', 'undertaking'],
        'technical': ['technical', 'technology', 'system', 'platform', 'software', 'hardware'],
        'business': ['business', 'commercial', 'market', 'customer', 'client'],
        'financial': ['financial', 'budget', 'cost', 'price', 'investment', 'roi'],
        'legal': ['legal', 'contract', 'agreement', 'terms', 'conditions', 'compliance'],
        'methodology': ['methodology', 'approach', 'process', 'framework', 'method'],
        'timeline': ['timeline', 'schedule', 'deadline', 'milestone', 'delivery'],
        'team': ['team', 'personnel', 'staff', 'resources', 'expertise'],
        'risk': ['risk', 'challenge', 'issue', 'mitigation', 'contingency'],
        'quality': ['quality', 'standard', 'requirement', 'specification'],
    }
    
    text_lower = text.lower()
    topics_found = []
    
    for topic_name, keywords in common_sections.items():
        # Count occurrences of topic keywords
        keyword_count = 0
        for keyword in keywords:
            keyword_count += len(re.findall(r'\b' + re.escape(keyword) + r'\b', text_lower))
        
        if keyword_count > 0:
            topics_found.append({
                "topic": topic_name,
                "keywords": keywords,
                "frequency": keyword_count,
                "confidence": min(keyword_count / 2, 1.0)  # Simple confidence score
            })
    
    # Sort by frequency and confidence
    topics_found.sort(key=lambda x: (x["frequency"], x["confidence"]), reverse=True)
    
    # Also extract unique nouns and important phrases
    additional_topics = extract_noun_phrases(text)
    
    # Combine and return top topics
    all_topics = [topic["topic"] for topic in topics_found[:max_topics]]
    all_topics.extend(additional_topics[:3])  # Add a few noun phrases
    
    return list(set(all_topics))[:max_topics]  # Remove duplicates

def extract_noun_phrases(text: str) -> List[str]:
    """
    Extract simple noun phrases from text (fallback without NLP library)
    """
    # Simple pattern-based noun phrase extraction
    sentences = re.split(r'[.!?]+', text)
    noun_phrases = []
    
    # Patterns for common noun phrases
    patterns = [
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',  # Capitalized phrases (proper nouns)
        r'\b(\w+ing\s+\w+)\b',                    # Gerund phrases
        r'\b(\w+\s+of\s+\w+)\b',                  # X of Y patterns
        r'\b(\w+\s+and\s+\w+)\b',                 # X and Y patterns
    ]
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 10:  # Skip very short sentences
            continue
            
        for pattern in patterns:
            matches = re.findall(pattern, sentence)
            for match in matches:
                if (len(match) > 5 and  # Reasonable length
                    match.lower() not in ['the', 'and', 'for', 'with', 'this', 'that'] and
                    not match[0].isdigit()):  # Not starting with number
                    noun_phrases.append(match)
    
    # Return unique phrases, most frequent first
    from collections import Counter
    phrase_counts = Counter(noun_phrases)
    return [phrase for phrase, count in phrase_counts.most_common(10)]

def classify_placeholder_type(placeholder: str) -> str:
    """
    Classify the type of placeholder
    """
    placeholder_lower = placeholder.lower()
    
    if re.match(r'_{3,}', placeholder):
        return "underscore_placeholder"
    elif placeholder.startswith('[') and placeholder.endswith(']'):
        return "bracket_placeholder" 
    elif placeholder.startswith('(') and placeholder.endswith(')'):
        return "parenthesis_placeholder"
    elif placeholder.startswith('<') and placeholder.endswith('>'):
        return "angle_bracket_placeholder"
    elif placeholder.startswith('{') and placeholder.endswith('}'):
        return "curly_brace_placeholder"
    elif any(word in placeholder_lower for word in ['todo', 'fill', 'insert', 'add']):
        return "instruction_placeholder"
    elif re.match(r'\?{2,}', placeholder):
        return "question_mark_placeholder"
    else:
        return "unknown_placeholder"
    

def identify_sections(text: str) -> List[dict]:
    """Identify document sections and headings"""
    sections = []
    lines = text.split('\n')
    
    current_section = None
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        # Detect section headers (simple heuristic)
        if (line.isupper() or 
            (len(line) < 100 and (line.endswith(':') or any(word in line.lower() for word in ['introduction', 'methodology', 'results', 'conclusion', 'abstract'])))):
            
            if current_section:
                sections.append(current_section)
            
            current_section = {
                "title": line,
                "content": "",
                "lineNumber": i,
                "needsCompletion": len(line.strip()) < 20  # Short headers likely need content
            }
        elif current_section:
            current_section["content"] += line + "\n"
    
    if current_section:
        sections.append(current_section)
    
    return sections

def find_completion_points(text: str) -> List[dict]:
    """Find places in the template that need completion"""
    completion_points = []
    
    # Look for common placeholder patterns
    placeholder_patterns = [
        r'_{3,}',  # ___ underscores
        r'\[.*?\]',  # [brackets]
        r'\(.*?\)',  # (parentheses)
        r'<.*?>',  # <angle brackets>
        r'XXX', r'TODO', r'FILL', r'INSERT'
    ]
    
    lines = text.split('\n')
    for i, line in enumerate(lines):
        for pattern in placeholder_patterns:
            matches = re.finditer(pattern, line)
            for match in matches:
                completion_points.append({
                    "lineNumber": i,
                    "position": match.start(),
                    "placeholder": match.group(),
                    "context": line,
                    "type": classify_placeholder_type(match.group())
                })
    
    # Also find very short sections that likely need expansion
    paragraphs = text.split('\n\n')
    for i, para in enumerate(paragraphs):
        if 0 < len(para.strip()) < 50:  # Very short paragraphs
            completion_points.append({
                "lineNumber": i,
                "position": 0,
                "placeholder": "SHORT_PARAGRAPH",
                "context": para,
                "type": "section_expansion"
            })
    
    return completion_points

def classify_document_type(text: str, filename: str) -> str:
    """Classify the type of document"""
    text_lower = text.lower()
    filename_lower = filename.lower()
    
    if any(word in text_lower + filename_lower for word in ['proposal', 'offer', 'quote']):
        return "proposal"
    elif any(word in text_lower + filename_lower for word in ['contract', 'agreement', 'legal']):
        return "contract"
    elif any(word in text_lower + filename_lower for word in ['report', 'analysis', 'findings']):
        return "report"
    elif any(word in text_lower + filename_lower for word in ['manual', 'guide', 'instructions']):
        return "manual"
    else:
        return "general"
    
