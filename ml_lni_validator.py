"""
ML-Based LNI Validator - Uses pre-trained Bibstyle-Detector model
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("WARNING: transformers not installed. Run: pip install transformers torch")


@dataclass
class LNIValidationResult:
    is_lni: bool
    confidence: float
    predicted_style: str
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    raw_text: str = ""


class LNIValidator:
    def __init__(self):
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers not installed. Run: pip install transformers torch")
        
        print("[LNI Validator] Loading pre-trained model...")
        self.pipe = pipeline("text-classification", model="PleIAs/Bibstyle-Detector")
        print("[LNI Validator] Model loaded successfully!")
    
    def validate(self, reference_text: str) -> LNIValidationResult:
        if not reference_text or not reference_text.strip():
            return LNIValidationResult(
                is_lni=False,
                confidence=0.0,
                predicted_style="unknown",
                issues=["Empty reference"],
                raw_text=reference_text
            )
        
        try:
            result = self.pipe(reference_text)[0]
            predicted_style = result['label']
            confidence = result['score']
            is_lni = predicted_style == 'lecture-notes-informatics'
            
            return LNIValidationResult(
                is_lni=is_lni,
                confidence=confidence,
                predicted_style=predicted_style,
                raw_text=reference_text
            )
        except Exception as e:
            return LNIValidationResult(
                is_lni=False,
                confidence=0.0,
                predicted_style="error",
                issues=[f"Validation error: {str(e)}"],
                raw_text=reference_text
            )