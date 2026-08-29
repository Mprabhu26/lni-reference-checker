"""
TIER 3: Professor Workflow Enhancements
Purpose: Prioritize professor's review workload based on confidence
Features:
  1. Confidence-based review prioritization
  2. Batch comparison mode (detect suspicious patterns across submission)
  3. Reference similarity detection (duplicate or related papers)
Version: 1.0
"""

from typing import List, Dict, Tuple, Optional
import re
from dataclasses import dataclass


@dataclass
class ReviewPriority:
    """Represents a reference that needs professor review"""
    key: str
    title: str
    reason: str
    confidence: float
    priority_score: float  # 0.0 (urgent) to 1.0 (can skip)
    evidence: List[str]    # List of red flags/concerns


def prioritize_for_review(
    verification_results: List,
) -> List[ReviewPriority]:
    """
    TIER 3 Feature 1: Confidence-Based Review Prioritization
    
    Returns references ordered by how much professor attention they need.
    Priority 0.0 = needs urgent review
    Priority 1.0 = can be skipped (high confidence)
    
    Professor sees HIGH PRIORITY first, then MODERATE, then LOW.
    Works with both VerificationResult objects and dictionaries.
    """
    priorities = []
    
    for result in verification_results:
        # Handle both dict and object attributes
        status = result.get("status") if isinstance(result, dict) else result.status
        confidence = result.get("confidence") if isinstance(result, dict) else result.confidence
        key = result.get("key") if isinstance(result, dict) else result.key
        title = result.get("title") if isinstance(result, dict) else result.title
        note = result.get("note") if isinstance(result, dict) else result.note
        is_retracted = result.get("is_retracted", False) if isinstance(result, dict) else getattr(result, "is_retracted", False)
        is_duplicate = result.get("is_duplicate", False) if isinstance(result, dict) else getattr(result, "is_duplicate", False)
        duplicate_of = result.get("duplicate_of") if isinstance(result, dict) else getattr(result, "duplicate_of", None)
        
        # Skip if this is a duplicate
        if is_duplicate:
            priority_score = 0.95
            priority_text = "DUPLICATE"
        # Skip already-verified with high confidence
        elif status == "verified" and confidence >= 0.92:
            priority_score = 1.0
            priority_text = "HIGH_CONFIDENCE_VERIFIED"
        
        # Medium priority: moderate confidence or partial matches
        elif status == "verified" and confidence >= 0.75:
            priority_score = 0.7
            priority_text = "MODERATE_CONFIDENCE_VERIFIED"
        
        # High priority: manual review needed
        elif status in ("manual_review", "suspicious"):
            if confidence < 0.4:
                priority_score = 0.1  # URGENT
                priority_text = "LOW_CONFIDENCE_SUSPICIOUS"
            elif confidence < 0.65:
                priority_score = 0.3
                priority_text = "QUESTIONABLE"
            else:
                priority_score = 0.6
                priority_text = "MODERATE_MANUAL_REVIEW"
        
        # Very high priority: likely fake
        elif status == "not_found":
            if confidence >= 0.85:
                priority_score = 0.05  # CRITICAL
                priority_text = "LIKELY_FAKE_HIGH_CONFIDENCE"
            else:
                priority_score = 0.2
                priority_text = "POSSIBLE_FAKE_LOW_CONFIDENCE"
        
        else:
            priority_score = 0.5
            priority_text = "OTHER"
        
        evidence = []
        if confidence < 0.5:
            evidence.append("Low confidence score")
        if status in ("manual_review", "suspicious", "not_found"):
            evidence.append(f"Status: {status}")
        if note:
            evidence.append(f"Note: {str(note)[:100]}")
        if is_retracted:
            evidence.append("PAPER IS RETRACTED")
        if is_duplicate:
            evidence.append(f"Duplicate of [{duplicate_of}]")
        
        priorities.append(ReviewPriority(
            key=key,
            title=str(title)[:60],
            reason=priority_text,
            confidence=confidence,
            priority_score=priority_score,
            evidence=evidence,
        ))
    
    # Sort by priority_score (low = urgent)
    priorities.sort(key=lambda p: (p.priority_score, -p.confidence))
    
    return priorities


def get_review_summary(
    priorities: List[ReviewPriority],
) -> Dict:
    """
    Summarize review priorities for professor.
    
    Returns:
    {
        "urgent": [...],        # priority_score < 0.2
        "important": [...],     # 0.2-0.6
        "optional": [...],      # 0.6-0.95
        "skip": [...],          # >= 0.95
        "total_to_review": N,
        "summary": "X papers need review (Y urgent)"
    }
    """
    urgent = [p for p in priorities if p.priority_score < 0.2]
    important = [p for p in priorities if 0.2 <= p.priority_score < 0.6]
    optional = [p for p in priorities if 0.6 <= p.priority_score < 0.95]
    skip = [p for p in priorities if p.priority_score >= 0.95]
    
    total_to_review = len(urgent) + len(important) + len(optional)
    
    summary = f"{total_to_review} paper(s) need review"
    if urgent:
        summary += f" ({len(urgent)} urgent)"
    
    return {
        "urgent": urgent,
        "important": important,
        "optional": optional,
        "skip": skip,
        "total_to_review": total_to_review,
        "summary": summary,
    }


def detect_batch_patterns(
    verification_results: List,
) -> Dict:
    """
    TIER 3 Feature 2: Batch Comparison Mode
    
    Detect suspicious patterns within a single submission:
    - References with metadata that differs from most others
    - Unusual year distribution
    - Similar/duplicate entries with slight variations
    - Metadata consistency anomalies
    
    Returns findings dict with warnings.
    Works with both VerificationResult objects and dictionaries.
    """
    if not verification_results:
        return {"patterns": [], "warnings": []}
    
    patterns = []
    warnings = []
    
    # Extract years for analysis
    years = []
    for r in verification_results:
        # Handle both dict and object
        title = r.get("title") if isinstance(r, dict) else getattr(r, "title", "")
        matched_title = r.get("matched_title") if isinstance(r, dict) else getattr(r, "matched_title", "")
        key = r.get("key") if isinstance(r, dict) else getattr(r, "key", "")
        
        title_text = matched_title or title or ""
        match = re.search(r'\b(19|20)\d{2}\b', title_text)
        if match:
            year = int(match.group())
            years.append((key, year))
    
    if years:
        year_values = [y for _, y in years]
        avg_year = sum(year_values) / len(year_values)
        min_year = min(year_values)
        max_year = max(year_values)
        
        # Flag references with years far from average
        if max_year - min_year > 20:
            for key, year in years:
                if year < avg_year - 15 or year > avg_year + 2:
                    patterns.append({
                        "type": "outlier_year",
                        "key": key,
                        "year": year,
                        "average": round(avg_year),
                        "message": f"Paper from {year} (mean: {round(avg_year)})"
                    })
    
    # Check for suspicious counts of references with same authors
    author_counts = {}
    for r in verification_results:
        # Handle both dict and object
        authors = r.get("correct_authors") if isinstance(r, dict) else getattr(r, "correct_authors", "")
        if authors:
            # Count first author
            first = str(authors).split(";")[0].strip().lower()
            if first:
                author_counts[first] = author_counts.get(first, 0) + 1
    
    for author, count in author_counts.items():
        if count >= 4:
            warnings.append({
                "type": "repeated_author",
                "author": author,
                "count": count,
                "message": f"Author appears in {count} references (could be self-plagiarism concern)"
            })
    
    # Check for similarity between entries (potential duplicates)
    for i, r1 in enumerate(verification_results):
        title1 = r1.get("title") if isinstance(r1, dict) else getattr(r1, "title", "")
        key1 = r1.get("key") if isinstance(r1, dict) else getattr(r1, "key", "")
        
        for r2 in verification_results[i+1:]:
            title2 = r2.get("title") if isinstance(r2, dict) else getattr(r2, "title", "")
            key2 = r2.get("key") if isinstance(r2, dict) else getattr(r2, "key", "")
            
            if title1 and title2:
                # Simple string similarity
                title1_words = set(str(title1).lower().split())
                title2_words = set(str(title2).lower().split())
                
                if title1_words and title2_words:
                    intersection = len(title1_words & title2_words)
                    union = len(title1_words | title2_words)
                    if union > 0:
                        similarity = intersection / union
                        if similarity > 0.75:  # 75%+ similar
                            patterns.append({
                                "type": "similar_titles",
                                "key1": key1,
                                "key2": key2,
                                "similarity": round(similarity * 100),
                                "message": f"[{key1}] and [{key2}] have {round(similarity*100)}% similar titles"
                            })
    
    return {
        "patterns": patterns,
        "warnings": warnings,
        "summary": f"Found {len(patterns)} patterns and {len(warnings)} warnings in batch"
    }


def find_similar_references(
    reference: Dict,
    candidate_pool: List[Dict],
) -> List[Dict]:
    """
    TIER 3 Feature 3: Reference Similarity Detection
    
    Given a reference, find similar ones in a pool (e.g., database, previous submissions).
    Used to detect:
    - Duplicate entries with slightly different metadata
    - Same paper cited multiple times
    - Similar papers by same author
    
    Returns sorted list of (similar_ref, similarity_score).
    """
    if not reference.get("title") or not candidate_pool:
        return []
    
    ref_title = reference.get("title", "").lower()
    ref_authors = reference.get("authors", "").lower()
    ref_year = reference.get("year", "")
    
    similarities = []
    
    for candidate in candidate_pool:
        cand_title = candidate.get("title", "").lower()
        cand_authors = candidate.get("authors", "").lower()
        cand_year = candidate.get("year", "")
        
        # Title similarity
        title_words_ref = set(ref_title.split())
        title_words_cand = set(cand_title.split())
        
        if not title_words_ref or not title_words_cand:
            continue
        
        title_similarity = len(title_words_ref & title_words_cand) / len(title_words_ref | title_words_cand)
        
        # Author similarity
        author_similarity = 0.0
        if ref_authors and cand_authors:
            authors_ref = set(a.strip().lower() for a in ref_authors.split(";"))
            authors_cand = set(a.strip().lower() for a in cand_authors.split(";"))
            if authors_ref or authors_cand:
                author_similarity = len(authors_ref & authors_cand) / max(len(authors_ref | authors_cand), 1)
        
        # Year match (exact or within 1 year)
        year_match = False
        if ref_year and cand_year:
            try:
                year_diff = abs(int(ref_year) - int(cand_year))
                year_match = year_diff <= 1
            except ValueError:
                pass
        
        # Combined score
        combined_score = (title_similarity * 0.5) + (author_similarity * 0.4)
        if year_match:
            combined_score += 0.1
        
        # Only include if reasonably similar
        if combined_score > 0.60:
            similarities.append({
                "candidate": candidate,
                "similarity_score": round(combined_score, 3),
                "title_sim": round(title_similarity, 3),
                "author_sim": round(author_similarity, 3),
                "year_match": year_match,
                "evidence": f"Title {round(title_similarity*100)}%, Authors {round(author_similarity*100)}%"
            })
    
    # Sort by similarity score
    similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    return similarities
