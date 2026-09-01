"""
Citation context analysis — checks if citations are actually used in paper.

Validates:
- Is a bibliography entry cited in the body text?
- Where and how many times is it cited?
- What is the citation context (surrounding text)?
- Are citations quality (substantial vs throwaway)?

NOTE: These are INFORMATIONAL CHECKS. They do not change REAL/SUSPICIOUS/FAKE verdicts.
"""

import re
from typing import Dict, List, Set, Tuple, Optional


def extract_citations_with_context(body: str, max_context_words: int = 50) -> Dict[str, List[str]]:
    """
    Extract citation keys from body text along with surrounding context.
    
    Returns: {
        'KEY1': ['...surrounding text...', '...another mention...'],
        'KEY2': ['...context...'],
    }
    """
    contexts = {}
    if not body:
        return contexts
    
    # Normalize citation markers in brackets to handle multiline breaks
    def normalize_brackets(m):
        content = m.group(1)
        normalized = re.sub(r'\s+', '', content)
        return '[' + normalized + ']'
    
    body_clean = re.sub(r'\[([A-Za-z0-9\s\n,;\-]+)\]', normalize_brackets, body)
    
    # Find all citations with surrounding context
    # Pattern: up to 50 words before, [CITATION], up to 50 words after
    for m in re.finditer(
        r'(.{0,200})(\[[A-Za-z][A-Za-z0-9+]{0,40}(?:[,;]\s*[A-Za-z][A-Za-z0-9+]{0,40})*\]|\[\d[\d,;\s\-]*\])(.{0,200})',
        body_clean,
    ):
        pre, cite, post = m.group(1), m.group(2), m.group(3)
        citation_text = f"...{pre}{cite}{post}...".strip()
        
        # Extract individual keys from potentially multi-key citations
        for key in re.split(r'\s*[,;]\s*', cite[1:-1]):
            key = key.strip()
            if key:
                if key not in contexts:
                    contexts[key] = []
                contexts[key].append(citation_text)
    
    # Also handle LaTeX \cite{} and numeric citations
    for m in re.finditer(r'\\(?:cite|citet|citep|Cite)\{([^}]+)\}', body_clean):
        pre_start = max(0, m.start() - 100)
        post_end = min(len(body_clean), m.end() + 100)
        pre = body_clean[pre_start:m.start()].strip()
        post = body_clean[m.end():post_end].strip()
        citation_text = f"...{pre}\\cite{{{m.group(1)}}}{post}...".strip()
        
        for key in m.group(1).split(','):
            key = key.strip()
            if key:
                if key not in contexts:
                    contexts[key] = []
                contexts[key].append(citation_text)
    
    return contexts


def detect_orphaned_entries(body: str, bibliography_keys: Set[str]) -> List[str]:
    """
    Find entries in bibliography that are never cited in body.
    Returns list of orphaned keys.
    """
    cited_keys = _extract_cited_keys(body)
    orphaned = []
    
    for key in bibliography_keys:
        # Check exact match and case-insensitive variants
        if key not in cited_keys and key.lower() not in {k.lower() for k in cited_keys}:
            orphaned.append(key)
    
    return orphaned


def detect_missing_citations(body: str, bibliography_keys: Set[str]) -> List[str]:
    """
    Find citations in body text that have no entry in bibliography.
    Returns list of missing citation keys.
    """
    cited_keys = _extract_cited_keys(body)
    bib_keys_lower = {k.lower(): k for k in bibliography_keys}
    
    missing = []
    for cited_key in cited_keys:
        cited_lower = cited_key.lower()
        if cited_lower not in bib_keys_lower:
            # Check if it's a numeric citation (not a key-based citation)
            if not re.fullmatch(r'__NUM_\d+__|__numeric_citations__', cited_key):
                missing.append(cited_key)
    
    return missing


def analyze_citation_quality(context_text: str, cited_entry_title: str) -> Dict:
    """
    Analyze the quality/importance of a citation based on context.
    
    Returns: {
        'quality_score': float (0-1),  # How substantial is this citation?
        'section': str,                 # Inferred section (intro, methods, etc)
        'is_primary': bool,             # Primary vs secondary reference?
        'reasoning': str
    }
    """
    score = 0.5  # Base score
    section = "unknown"
    is_primary = True
    reasoning = []
    
    # Check section keywords
    if re.search(r'\b(?:introduction|related work|background|preliminaries|overview)\b', context_text, re.IGNORECASE):
        section = "introduction"
        score += 0.1  # Intro citations often fundamental
        reasoning.append("Cited in introduction/related work section")
    elif re.search(r'\b(?:method|approach|algorithm|technique|framework)\b', context_text, re.IGNORECASE):
        section = "methods"
        score += 0.2  # Method citations are primary
        reasoning.append("Cited in methods/algorithm section")
    elif re.search(r'\b(?:experiment|result|evaluation|finding|outcome)\b', context_text, re.IGNORECASE):
        section = "results"
        score += 0.15
        reasoning.append("Cited in results section")
    elif re.search(r'\b(?:discussion|conclusion|future work)\b', context_text, re.IGNORECASE):
        section = "conclusion"
        score += 0.05
        reasoning.append("Cited in conclusion/discussion")
    elif re.search(r'\b(?:footnote|note|aside|remark)\b', context_text, re.IGNORECASE):
        section = "footnote"
        score -= 0.2  # Footnotes are throwaway
        is_primary = False
        reasoning.append("Cited in footnote (may be tangential)")
    
    # Check citation depth (how substantive?)
    if re.search(r'\b(?:extensively|thoroughly|carefully|in detail|rigorously|fundamentally)\b', context_text, re.IGNORECASE):
        score += 0.2
        reasoning.append("Citation described as extensive/thorough")
    
    if re.search(r'\b(?:briefly|simply|note that|we note|as noted|cf\.|e\.g\.?|for example)\b', context_text, re.IGNORECASE):
        score -= 0.15
        is_primary = False
        reasoning.append("Citation appears incidental (e.g., cf., as noted)")
    
    # Check if cited for novelty or evaluation
    if re.search(r'\b(?:novel|new|proposed|our|we|demonstrate|show|prove|improve|better|outperform)\b', context_text, re.IGNORECASE):
        score += 0.1
        reasoning.append("Cited in context of novel contribution")
    
    # Check if comparing or contrasting
    if re.search(r'\b(?:compare|contrast|differ|unlike|in contrast to|versus|vs\.?)\b', context_text, re.IGNORECASE):
        score += 0.15
        reasoning.append("Citation used for comparison/contrast")
    
    # Check title matching (does citation context match paper title?)
    if cited_entry_title:
        context_words = set(re.findall(r'\b\w{3,}\b', context_text.lower()))
        title_words = set(re.findall(r'\b\w{3,}\b', cited_entry_title.lower()))
        overlap = len(context_words & title_words) / max(len(title_words), 1)
        if overlap > 0.4:
            score += 0.1
            reasoning.append(f"Citation context matches paper topic ({int(overlap*100)}% word overlap)")
        elif overlap < 0.1:
            score -= 0.1
            reasoning.append("Citation context poorly related to cited paper's title")
    
    # Clamp score to [0, 1]
    score = max(0.0, min(1.0, score))
    
    return {
        'quality_score': round(score, 3),
        'section': section,
        'is_primary': is_primary,
        'reasoning': '; '.join(reasoning) if reasoning else 'Citation context analyzed'
    }


def detect_citation_chains(body: str) -> List[Dict]:
    """
    Detect second-order or higher citations (citing what was cited).
    
    Example: "As Smith et al. [X] note from Jones's work [Y]..."
    This indicates a chain: Body text → [X] → [Y]
    
    Returns list of potential citation chains with details.
    """
    chains = []
    
    # Pattern: Author names + paper or work + followed by citation
    chain_pattern = r'(?:According to|As noted by|As shown in|Based on|From)\s+([A-Z][A-Za-z\-]+(?:\s+et\s+al\.?)?)\s+(?:\[\w+\]|["\'][^"\']{2,50}["\'])\s+(?:in|citing|from|quoting)\s+([A-Za-z\-]+["\']?s?)\s+\[\w+\]'
    
    for m in re.finditer(chain_pattern, body, re.IGNORECASE):
        chains.append({
            'pattern': 'author_chain',
            'first_author': m.group(1),
            'second_author': m.group(2),
            'context': body[max(0, m.start()-50):min(len(body), m.end()+50)],
            'confidence': 'low',  # User should verify
            'note': 'Second-order citation detected; verify that both papers are actually in bibliography'
        })
    
    return chains


def generate_citation_report(body: str, bibliography_keys: Set[str], citation_contexts: Dict) -> Dict:
    """
    Generate comprehensive citation analysis report.
    
    Returns: {
        'total_bibliography_entries': int,
        'total_unique_citations': int,
        'orphaned_entries': [...],
        'missing_citations': [...],
        'citation_density': dict,  # Per-entry stats
        'citation_chains': [...],
        'warnings': [...]
    }
    """
    cited_keys = _extract_cited_keys(body)
    orphaned = detect_orphaned_entries(body, bibliography_keys)
    missing = detect_missing_citations(body, bibliography_keys)
    chains = detect_citation_chains(body)
    
    warnings = []
    
    # Orphaned entries (in bib but not cited)
    if orphaned:
        warnings.append({
            'type': 'orphaned_entries',
            'severity': 'warn',
            'count': len(orphaned),
            'entries': orphaned[:10],  # Show first 10
            'message': f'{len(orphaned)} bibliography entries are never cited in the paper body'
        })
    
    # Missing citations (cited but not in bib)
    if missing:
        warnings.append({
            'type': 'missing_citations',
            'severity': 'error',
            'count': len(missing),
            'entries': missing[:10],  # Show first 10
            'message': f'{len(missing)} citations in body have no bibliography entry'
        })
    
    # Citation chains
    if chains:
        warnings.append({
            'type': 'citation_chains',
            'severity': 'info',
            'count': len(chains),
            'entries': chains,
            'message': f'{len(chains)} potential second-order citations detected; verify authenticity'
        })
    
    # Citation density analysis
    citation_density = {}
    for key in bibliography_keys:
        key_lower = key.lower()
        contexts = citation_contexts.get(key, [])
        count = len(contexts)
        
        citation_density[key] = {
            'citation_count': count,
            'is_cited': count > 0,
            'contexts': contexts[:3],  # First 3 contexts
        }
        
        if count > 5:
            warnings.append({
                'type': 'over_citation',
                'severity': 'info',
                'entry': key,
                'count': count,
                'message': f'Reference [{key}] is cited {count} times (unusually frequent)'
            })
    
    return {
        'total_bibliography_entries': len(bibliography_keys),
        'total_unique_citations': len(set(k.lower() for k in cited_keys)),
        'orphaned_entries': orphaned,
        'missing_citations': missing,
        'citation_density': citation_density,
        'citation_chains': chains,
        'warnings': warnings
    }


# ============================================================================
# Helper functions
# ============================================================================

def _extract_cited_keys(body: str) -> Set[str]:
    """Extract all citation keys from body text (used by other functions)."""
    keys = set()
    if not body:
        return keys
    
    # Normalize brackets
    body_clean = re.sub(r'\[([A-Za-z0-9\s\n,;\-]+)\]', 
                       lambda m: '[' + re.sub(r'\s+', '', m.group(1)) + ']', 
                       body)
    
    # LNI format [KEY], [KEY1, KEY2], [KEY1a]
    for m in re.finditer(r'\[([^\]]+)\]', body_clean):
        prefix = body_clean[max(0, m.start() - 12):m.start()].lower()
        if re.search(r'(?:e\.g\.?|z\.b\.?|cf\.?)\s*$', prefix):
            continue
        for k in re.split(r'\s*[,;]\s*', m.group(1)):
            k = k.strip()
            if re.fullmatch(r'\d+', k):
                keys.add(f'__NUM_{k}__')
                keys.add('__numeric_citations__')
            elif re.fullmatch(r'[A-Z][A-Za-z+]{0,5}\d{2}[a-z]?', k):
                keys.add(k)
    
    # LaTeX \cite{}
    for m in re.finditer(r'\\(?:cite|citet|citep|Cite)\{([^}]+)\}', body_clean):
        for k in m.group(1).split(','):
            k = k.strip()
            if k:
                keys.add(k)
    
    return keys
