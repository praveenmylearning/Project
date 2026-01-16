"""Summarization Module - Output formatting and generation"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from config.schemas import create_summarizer_output, create_error_response, ComponentType

logger = logging.getLogger(__name__)


class SummarizerModule:
    """
    Phase 5: Generate final output with formatting and citations
    
    Features:
    - Result synthesis
    - Key insights extraction
    - Citation generation
    - Financial formatting
    """
    
    def __init__(self, prompt_path: Optional[Path] = None):
        self.prompt_path = prompt_path
        self.prompt = self._load_prompt() if prompt_path else ""
    
    def _load_prompt(self) -> str:
        """Load summarizer prompt"""
        try:
            with open(self.prompt_path, 'r') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to load prompt: {e}")
            return ""
    
    async def run(self, execution_output: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """
        Generate final summary
        
        Args:
            execution_output: Output from execution module
            context: Context manager
            
        Returns:
            Summarization output with formatted results
        """
        logger.info("Summarizer generating final output...")
        
        try:
            # Extract results
            results = execution_output.get("data", {}).get("results", {})
            
            # Generate summary
            summary = self._generate_summary(results)
            
            # Extract key insights
            key_insights = self._extract_insights(results)
            
            # Generate citations
            citations = self._generate_citations(context)
            
            output = create_summarizer_output(
                summary=summary,
                key_insights=key_insights,
                citations=citations,
                confidence=0.85
            )
            
            context.add_global("summarizer_output", output)
            context.mark_step_completed("summarization")
            logger.info("Summarization complete")
            
            return output
        
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            context.record_error("summarization", str(e))
            return create_error_response(
                ComponentType.SUMMARIZATION,
                "SUMMARIZATION_ERROR",
                f"Failed to generate summary: {str(e)}",
                recoverable=True
            )
    
    def _generate_summary(self, results: Dict[str, Any]) -> str:
        """Generate summary from results"""
        summary_parts = []
        
        for key, value in results.items():
            if isinstance(value, (int, float)):
                summary_parts.append(f"{key}: {value}")
        
        return " | ".join(summary_parts) if summary_parts else "Analysis complete."
    
    def _extract_insights(self, results: Dict[str, Any]) -> List[str]:
        """Extract key insights from results"""
        insights = []
        
        if "pe_ratio" in results:
            pe = results["pe_ratio"]
            if pe < 15:
                insights.append("Stock appears undervalued based on P/E ratio")
            elif pe > 25:
                insights.append("Stock appears relatively expensive")
        
        if "roe" in results:
            roe = results["roe"]
            if roe > 0.15:
                insights.append("Strong return on equity indicates good profitability")
        
        return insights if insights else ["Analysis completed successfully"]
    
    def _generate_citations(self, context: Any) -> List[Dict[str, str]]:
        """Generate citations from retrieved documents"""
        retrieval_output = context.get_global("retrieval_output")
        
        citations = []
        if retrieval_output and "data" in retrieval_output:
            documents = retrieval_output["data"].get("documents", [])
            for doc in documents[:3]:
                citations.append({
                    "title": doc.get("title", "Document"),
                    "source": doc.get("id", "unknown")
                })
        
        return citations
