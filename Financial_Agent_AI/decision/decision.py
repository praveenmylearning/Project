"""
Decision Module - Plan execution and select tools/methods
Uses centralized config and standardized JSON schema
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
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

# Standard schema for decision output
DECISION_SCHEMA = {
    "status": "success",
    "data": {
        "execution_plan": {
            "primary_method": "",
            "steps": []
        },
        "execution_graph": {
            "nodes": [],
            "edges": []
        },
        "tool_parameters": {},
        "alternative_paths": [],
        "confidence": 0.0,
        "next_step": "execution",
        "reasoning": ""
    },
    "metadata": {
        "component": "decision",
        "timestamp": "",
        "version": "1.0"
    }
}


class DecisionModule:
    """
    Creates execution plans based on perception output:
    - Selects appropriate tools/methods
    - Sets parameters for each tool
    - Plans execution sequence
    - Provides alternative paths
    - Handles tool failures
    
    Returns standardized execution plan JSON
    """
    
    def __init__(self, prompt_path: Optional[Path] = None):
        """Initialize decision module with prompt"""
        self.prompt_path = prompt_path or Path("prompts/decision_prompt.txt")
        self.prompt = self._load_prompt()
        self.llm = get_llm_provider()
        self.failed_tools: Set[str] = set()
        self.tool_retry_counts: Dict[str, int] = {}
        self.max_retries = 3
        
    def _load_prompt(self) -> str:
        """Load decision prompt from file"""
        try:
            with open(self.prompt_path, 'r') as f:
                content = f.read()
                logger.info(f"Loaded decision prompt from {self.prompt_path}")
                return content
        except Exception as e:
            logger.error(f"Failed to load decision prompt: {e}")
            return ""
    
    async def run(self, perception_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate execution plan based on perception output
        
        Args:
            perception_output: Output from perception module (JSON schema)
            
        Returns:
            Decision output with execution plan (JSON schema)
        """
        logger.info("Decision module generating execution plan...")
        
        try:
            # Step 1: Extract perception data
            intent = perception_output.get("data", {}).get("interpreted_intent", "general")
            entities = perception_output.get("data", {}).get("entities", {})
            confidence = perception_output.get("data", {}).get("confidence", 0.7)
            
            # Step 2: Format prompt for LLM
            system_prompt = self.prompt
            perception_json = format_json_for_llm(perception_output)
            user_prompt = f"""
            Based on this perception output, create an execution plan:
            
            {perception_json}
            
            Determine:
            1. Which tools to use
            2. What parameters to set
            3. What order to execute
            4. Alternative paths if tools fail
            
            Provide output in valid JSON format.
            """
            
            # Step 3: Call LLM
            response = self.llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,  # Low for structured output
                max_tokens=3000
            )
            
            # Step 4: Parse JSON response
            decision_output = parse_llm_json(response)
            
            # Step 5: Validate and enhance
            if not decision_output or "data" not in decision_output:
                decision_output = self._create_default_plan(intent)
            
            # Step 6: Add metadata
            decision_output["metadata"]["timestamp"] = datetime.utcnow().isoformat() + "Z"
            
            logger.info(f"Decision identified method: {decision_output['data'].get('execution_plan', {}).get('primary_method', 'unknown')}")
            return decision_output
            
        except Exception as e:
            logger.error(f"Decision module error: {e}")
            return self._create_error_response(str(e))
    
    def _create_default_plan(self, intent: str) -> Dict[str, Any]:
        """Create default execution plan based on intent"""
        output = json.loads(json.dumps(DECISION_SCHEMA))  # Deep copy
        
        # Determine method based on intent
        intent_methods = {
            "valuation": "DCF",
            "analysis": "Financial Ratio Analysis",
            "comparison": "Multi-Company Comparison",
            "research": "Document Retrieval",
            "earnings": "Earnings Analysis",
            "risk": "Risk Assessment",
            "general": "General Analysis"
        }
        
        method = intent_methods.get(intent, "General Analysis")
        
        # Create basic execution steps
        output["data"]["execution_plan"]["primary_method"] = method
        output["data"]["execution_plan"]["steps"] = [
            {
                "step_number": 1,
                "tool": "RAG Pipeline",
                "task": "Retrieve company financials",
                "parameters": {}
            },
            {
                "step_number": 2,
                "tool": "FinanceTools" if method == "DCF" else "Analysis Tools",
                "task": f"Perform {method}",
                "parameters": {}
            }
        ]
        
        output["data"]["execution_graph"]["nodes"] = [
            {"id": 1, "tool": "RAG Pipeline", "parallel": False},
            {"id": 2, "tool": method, "parallel": False}
        ]
        
        output["data"]["execution_graph"]["edges"] = [
            {"from": 1, "to": 2, "dependency": "data"}
        ]
        
        output["data"]["confidence"] = 0.7
        output["data"]["reasoning"] = f"Selected {method} for {intent}"
        
        return output
    
    def _create_error_response(self, error: str) -> Dict[str, Any]:
        """Create error response"""
        output = json.loads(json.dumps(DECISION_SCHEMA))
        output["status"] = "error"
        output["data"]["reasoning"] = f"Error during decision: {error}"
        output["data"]["confidence"] = 0.0
        return output


# Convenience function
async def decide_execution(perception_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to create execution plan
    
    Usage:
        plan = await decide_execution(perception_output)
    """
    module = DecisionModule()
    return await module.run(perception_output)
