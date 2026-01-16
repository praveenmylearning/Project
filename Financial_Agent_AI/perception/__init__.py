"""Perception Module - Query understanding and routing"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
from config.schemas import create_perception_output, ComponentType, create_error_response

logger = logging.getLogger(__name__)


class PerceptionModule:
    """
    Phase 1: Understand user query and decide routing strategy
    
    Outputs:
    - Interpreted intent (equity_valuation, financial_analysis, etc.)
    - Route decision (agentic, retrieval, hybrid)
    - Extracted entities
    - Confidence score
    """
    
    def __init__(self, prompt_path: Optional[Path] = None):
        self.prompt_path = prompt_path
        self.prompt = self._load_prompt() if prompt_path else ""
    
    def _load_prompt(self) -> str:
        """Load perception prompt"""
        try:
            with open(self.prompt_path, 'r') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to load prompt: {e}")
            return ""
    
    async def run(self, query: str, context: Any) -> Dict[str, Any]:
        """
        Analyze query and determine routing
        
        Args:
            query: User query
            context: Context manager
            
        Returns:
            Perception output with intent, route, and confidence
        """
        logger.info(f"Perception analyzing: {query[:50]}...")
        
        try:
            # Extract intent from query
            intent = self._extract_intent(query)
            route = self._decide_route(query, intent)
            entities = self._extract_entities(query)
            
            output = create_perception_output(
                interpreted_intent=intent,
                route=route,
                entities=entities,
                confidence=0.85,
                reasoning=f"Query routed to {route} based on {intent}"
            )
            
            context.add_global("perception_output", output)
            context.mark_step_completed("perception")
            logger.info(f"Perception complete: intent={intent}, route={route}")
            
            return output
        
        except Exception as e:
            logger.error(f"Perception error: {e}")
            context.record_error("perception", str(e))
            return create_error_response(
                ComponentType.PERCEPTION,
                "ANALYSIS_ERROR",
                f"Failed to analyze query: {str(e)}",
                recoverable=True
            )
    
    def _extract_intent(self, query: str) -> str:
        """Extract intent from query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["dcf", "valuation", "intrinsic", "fair value"]):
            return "equity_valuation"
        elif any(word in query_lower for word in ["compare", "vs", "versus", "difference"]):
            return "financial_comparison"
        elif any(word in query_lower for word in ["analyze", "analysis", "metric", "ratio"]):
            return "financial_analysis"
        else:
            return "general_financial_analysis"
    
    def _decide_route(self, query: str, intent: str) -> str:
        """Decide execution route"""
        query_lower = query.lower()
        
        # If query has specific document/file references, use retrieval
        if any(word in query_lower for word in ["file", "document", "pdf", "report", "filing"]):
            return "retrieval"
        
        # If complex analysis needed, use hybrid
        if intent in ["financial_comparison", "equity_valuation"]:
            return "hybrid"
        
        # Default to agentic for simple queries
        return "agentic"
    
    def _extract_entities(self, query: str) -> list:
        """Extract financial entities from query"""
        # Simple extraction - can be enhanced with NER
        entities = []
        
        # Company names (simple list)
        companies = ["apple", "microsoft", "tesla", "google", "amazon", "meta", "nvidia"]
        for company in companies:
            if company in query.lower():
                entities.append(company)
        
        return entities
