"""
Summarization Module - Format analysis results into professional reports
Uses centralized config and standardized JSON schema
"""

import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
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

# Standard schema for summarization output
SUMMARIZATION_SCHEMA = {
    "status": "success",
    "data": {
        "summary": "",
        "key_insights": [],
        "detailed_analysis": "",
        "recommendations": {},
        "citations": [],
        "confidence": 0.0,
        "next_questions": [],
        "reasoning": ""
    },
    "metadata": {
        "component": "summarization",
        "timestamp": "",
        "version": "1.0"
    }
}


class SummarizerModule:
    """
    Synthesizes analysis results into professional, formatted output.
    
    Responsibilities:
    - Format financial results professionally
    - Apply financial number formatting ($1.23B, 12.34%)
    - Create tables and structured output
    - Add citations and confidence levels
    - Generate recommendations
    - Suggest follow-up questions
    """
    
    def __init__(self, prompt_path: Optional[Path] = None):
        """Initialize summarizer module with prompt"""
        self.prompt_path = prompt_path or Path("prompts/summarizer_prompt.txt")
        self.prompt = self._load_prompt()
        self.llm = get_llm_provider()
        self.execution_history = []
        
    def _load_prompt(self) -> str:
        """Load summarizer prompt from file"""
        try:
            with open(self.prompt_path, 'r') as f:
                content = f.read()
                logger.info(f"Loaded summarizer prompt from {self.prompt_path}")
                return content
        except Exception as e:
            logger.error(f"Failed to load summarizer prompt: {e}")
            return self._get_default_prompt()
    
    def _get_default_prompt(self) -> str:
        """Fallback prompt if file not found"""
        return """You are a financial report formatter.

Your task is to transform analysis results into a professional summary:
1. Create executive summary (1-2 sentences)
2. List key insights
3. Add detailed analysis
4. Include recommendations
5. Add citations
6. Suggest follow-up questions

Output JSON:
{
  "summary": "executive summary",
  "key_insights": ["insight 1", "insight 2"],
  "detailed_analysis": "detailed breakdown",
  "recommendations": {"action": "BUY/HOLD/SELL", "reasoning": "..."},
  "citations": [{"source": "...", "page": 1}],
  "confidence": 0.85,
  "next_questions": ["question 1", "question 2"]
}"""
    
    async def run(self, execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution: synthesize and format analysis results.
        
        Args:
            execution_results: Output from execution module
            
        Returns:
            Standardized summarization output
        """
        try:
            # Prepare LLM prompt
            user_prompt = f"""Format this financial analysis into a professional summary:

Analysis Results: {json.dumps(execution_results, indent=2)}

Create a comprehensive but concise report with:
1. Executive summary
2. Key insights
3. Detailed analysis
4. Recommendations
5. Citations and assumptions
6. Suggested follow-up questions

Respond with ONLY valid JSON (no markdown, no extra text)."""

            # Call LLM
            response = self.llm.generate(
                prompt=user_prompt,
                system_prompt=self.prompt,
                temperature=0.2  # Low temp for consistent formatting
            )
            
            # Parse response
            parsed = parse_llm_json(response)
            
            # Validate minimum required fields
            if not all(k in parsed for k in ["summary", "key_insights", "confidence"]):
                return self._create_default_output(execution_results)
            
            # Create output with schema
            output = {
                "status": "success",
                "data": {
                    "summary": parsed.get("summary", ""),
                    "key_insights": parsed.get("key_insights", []),
                    "detailed_analysis": parsed.get("detailed_analysis", ""),
                    "recommendations": parsed.get("recommendations", {}),
                    "citations": parsed.get("citations", []),
                    "confidence": float(parsed.get("confidence", 0.6)),
                    "next_questions": parsed.get("next_questions", []),
                    "reasoning": parsed.get("reasoning", "")
                },
                "metadata": {
                    "component": "summarization",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "version": "1.0"
                }
            }
            
            self.execution_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "status": "success",
                "output": output
            })
            
            return output
            
        except json.JSONDecodeError as e:
            return self._create_error_response(f"JSON parse error: {str(e)}")
        except Exception as e:
            logger.error(f"ERROR in summarization: {e}")
            return self._create_default_output(execution_results)
    
    def _create_default_output(self, execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create fallback output using template formatting"""
        results = execution_results.get("data", {}).get("results", {})
        
        # Extract key information
        summary = f"Financial analysis completed"
        if "intrinsic_value" in results:
            summary = f"Intrinsic value calculated at ${results['intrinsic_value']:.2f}"
        
        key_insights = []
        if "valuation_range" in results:
            key_insights.append(f"Valuation range: ${results['valuation_range'][0]:.2f} - ${results['valuation_range'][1]:.2f}")
        if "confidence" in results:
            key_insights.append(f"Analysis confidence: {results.get('confidence', 0.85):.1%}")
        
        output = {
            "status": "success",
            "data": {
                "summary": summary,
                "key_insights": key_insights or ["Analysis completed successfully"],
                "detailed_analysis": f"Detailed analysis based on execution results",
                "recommendations": {
                    "action": "REVIEW",
                    "reasoning": "Fallback output generated from execution results"
                },
                "citations": [],
                "confidence": float(execution_results.get("data", {}).get("confidence", 0.75)),
                "next_questions": [
                    "Would you like to explore different scenarios?",
                    "What are your risk parameters?"
                ],
                "reasoning": "Fallback formatting applied"
            },
            "metadata": {
                "component": "summarization",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "version": "1.0"
            }
        }
        
        self.execution_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "status": "fallback",
            "output": output
        })
        
        return output
    
    def _create_error_response(self, error_msg: str) -> Dict[str, Any]:
        """Create error response"""
        return {
            "status": "error",
            "data": {
                "error_type": "formatting_error",
                "error_message": error_msg,
                "recoverable": True,
                "suggested_recovery": "Try with simpler analysis results"
            },
            "metadata": {
                "component": "summarization",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "version": "1.0"
            }
        }


async def summarize_results(execution_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Async convenience function to summarize analysis results.
    
    Usage:
        output = await summarize_results(execution_output)
    """
    module = SummarizerModule()
    return await module.run(execution_results)


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        sample_results = {
            "status": "success",
            "data": {
                "results": {
                    "intrinsic_value": 175.50,
                    "current_price": 152.00,
                    "upside": 0.152,
                    "valuation_range": [160, 190],
                    "confidence": 0.88
                },
                "tool_used": "calculate_dcf"
            }
        }
        
        output = await summarize_results(sample_results)
        print(json.dumps(output, indent=2))
    
    asyncio.run(main())
