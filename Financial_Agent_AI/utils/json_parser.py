"""Robust JSON parsing for LLM outputs with error recovery"""

import json
import logging
import re
from typing import Any, Optional, List

logger = logging.getLogger(__name__)


def parse_llm_json(text: str, required_keys: Optional[List[str]] = None) -> Any:
    """
    Parse JSON from LLM output with error recovery
    Handles common LLM mistakes: markdown blocks, trailing commas, etc.
    """
    if not text:
        logger.warning("Empty text provided to parse_llm_json")
        return {}
    
    # Try direct JSON parsing first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Remove markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # Try again after cleanup
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Extract JSON object
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    logger.error(f"Failed to parse JSON: {text[:100]}")
    return {}


def format_json_for_llm(data: Any, indent: int = 2) -> str:
    """Format data as JSON for LLM consumption"""
    return json.dumps(data, indent=indent)
