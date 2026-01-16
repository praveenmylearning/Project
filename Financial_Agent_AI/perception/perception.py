"""
Perception Module - Analyze financial queries and extract intent
Uses centralized config and standardized JSON schema
"""

import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Import centralized config
from config.environment import config
from config.llm_config import get_llm_provider

# Import utilities
try:
    from utils.json_parser import parse_llm_json, format_json_for_llm
except ImportError:
    def parse_llm_json(text, required_keys=None):
        try:
            return json.loads(text)
        except:
            return {}
    def format_json_for_llm(data):
        return json.dumps(data, indent=2)

logger = logging.getLogger(__name__)

# Standard schema for perception output
PERCEPTION_SCHEMA = {
    "status": "success",
    "data": {
        "interpreted_intent": "",
        "entities": {
            "companies": [],
            "metrics": [],
            "time_period": "",
            "fiscal_year": None
        },
        "required_analysis": [],
        "data_sources_needed": [],
        "confidence": 0.0,
        "next_step": "decision",
        "reasoning": ""
    },
    "metadata": {
        "component": "perception",
        "timestamp": "",
        "version": "1.0"
    }
}


class PerceptionModule:
    """
    Analyzes financial queries to extract:
    - Intent (what user wants)
    - Entities (companies, metrics, dates)
    - Route (valuation, analysis, comparison, etc)
    - Data requirements
    
    Returns standardized JSON schema for downstream modules
    """
    
    def __init__(self, prompt_path: Optional[Path] = None):
        """Initialize perception module with prompt"""
        self.prompt_path = prompt_path or Path("prompts/perception_prompt.txt")
        self.prompt = self._load_prompt()
        self.llm = get_llm_provider()
        
    def _load_prompt(self) -> str:
        """Load perception prompt from file"""
        try:
            with open(self.prompt_path, 'r') as f:
                content = f.read()
                logger.info(f"Loaded perception prompt from {self.prompt_path}")
                return content
        except Exception as e:
            logger.error(f"Failed to load perception prompt: {e}")
            return ""
    
    async def run(self, query: str) -> Dict[str, Any]:
        """
        Analyze query and extract financial intent
        
        Args:
            query: User's financial query
            
        Returns:
            Standardized perception output JSON
        """
        logger.info(f"Perception analyzing query: {query[:100]}...")
        
        try:
            # Step 1: Format prompt for LLM
            system_prompt = self.prompt
            user_prompt = f"""
            Analyze this financial query and extract structured information:
            
            Query: {query}
            
            Provide output in valid JSON format matching the schema provided.
            """
            
            # Step 2: Call LLM
            response = self.llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,  # Low temperature for structured output
                max_tokens=2000
            )
            
            # Step 3: Parse JSON response
            perception_output = parse_llm_json(response)
            
            # Step 4: Validate and enhance output
            if not perception_output or "data" not in perception_output:
                perception_output = self._create_default_output(query)
            
            # Step 5: Add metadata
            perception_output["metadata"]["timestamp"] = datetime.utcnow().isoformat() + "Z"
            
            logger.info(f"Perception identified intent: {perception_output['data'].get('interpreted_intent', 'unknown')}")
            return perception_output
            
        except Exception as e:
            logger.error(f"Perception module error: {e}")
            return self._create_error_response(str(e))
    
    def _create_default_output(self, query: str) -> Dict[str, Any]:
        """Create default output for unparseable responses"""
        output = json.loads(json.dumps(PERCEPTION_SCHEMA))  # Deep copy
        
        # Try to identify intent from keywords
        intent_keywords = {
            "valuation": ["value", "intrinsic", "dcf", "fair value", "price target"],
            "analysis": ["analyze", "analysis", "performance", "financial health"],
            "comparison": ["compare", "vs", "versus", "vs.", "similar"],
            "research": ["research", "find", "tell me", "what", "about"],
            "earnings": ["earnings", "eps", "profit", "income"],
            "risk": ["risk", "risks", "risky", "exposure", "threat"],
        }
        
        query_lower = query.lower()
        intent = "general"
        for key, keywords in intent_keywords.items():
            if any(kw in query_lower for kw in keywords):
                intent = key
                break
        
        output["data"]["interpreted_intent"] = intent
        output["data"]["entities"]["time_period"] = "Latest"
        output["data"]["confidence"] = 0.6
        output["data"]["reasoning"] = f"Identified intent '{intent}' from query keywords"
        
        return output
    
    def _create_error_response(self, error: str) -> Dict[str, Any]:
        """Create error response"""
        output = json.loads(json.dumps(PERCEPTION_SCHEMA))
        output["status"] = "error"
        output["data"]["reasoning"] = f"Error during perception: {error}"
        output["data"]["confidence"] = 0.0
        return output


# Convenience function
async def perceive_query(query: str) -> Dict[str, Any]:
    """
    Convenience function to analyze a query
    
    Usage:
        output = await perceive_query("What's Apple's intrinsic value?")
    """
    module = PerceptionModule()
    return await module.run(query)
