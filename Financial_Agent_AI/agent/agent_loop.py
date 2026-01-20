"""
Agent Loop - 5-Phase Orchestration Engine for Single-Agent System
Implements: Perception → Decision → Retrieval → Execution → Summarization

Coordinates a single planning agent through sequential phases of analysis.
Uses centralized configuration from config/environment.py
"""

import logging
import asyncio
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from perception import PerceptionModule
from decision import DecisionModule
from retrieval import RAGPipeline
from action import CodeExecutor
from summarization import SummarizerModule
from agent.contextManager import ContextManager
from config.schemas import create_error_response, ComponentType, StatusType
from config.environment import config  # Centralized configuration

logger = logging.getLogger(__name__)

# Load profiles configuration
_profile_path = Path(__file__).parent.parent / "config" / "profiles.yaml"
if _profile_path.exists():
    with open(_profile_path) as f:
        PROFILES = yaml.safe_load(f)
else:
    PROFILES = {}


class AgentLoop:
    """
    Core orchestration engine for financial analysis
    
    Single-agent 5-phase agentic reasoning pipeline:
    1. Perception: Analyze query intent and decide execution strategy
    2. Decision: Build execution graph and plan next actions
    3. Retrieval: Fetch relevant documents if needed
    4. Execution: Run financial calculations and analysis
    5. Summarization: Format results with financial insights
    
    Features:
    - Unified JSON schema communication between components
    - Memory-based failure adaptation
    - Graph-based execution planning
    - Flexible execution modes (agentic, retrieval, hybrid)
    - Error recovery with fallback strategies
    - Session memory for multi-turn conversations
    """
    
    def __init__(
        self,
        perception_prompt: Optional[str] = None,
        decision_prompt: Optional[str] = None,
        retrieval_prompt: Optional[str] = None,
        summarizer_prompt: Optional[str] = None,
        mode: str = "hybrid",
        strategy: str = "exploratory"
    ):
        """Initialize agent loop with all components"""
        self.mode = mode
        self.strategy = strategy
        self.start_time = datetime.now()
        
        # Initialize components (Phase 1-5)
        self.perception = PerceptionModule(Path(perception_prompt) if perception_prompt else None)
        self.decision = DecisionModule(Path(decision_prompt) if decision_prompt else None)
        self.retrieval = RAGPipeline(Path(retrieval_prompt) if retrieval_prompt else None)
        self.executor = CodeExecutor()
        self.summarizer = SummarizerModule(Path(summarizer_prompt) if summarizer_prompt else None)
        self.context_manager = ContextManager()
        
        logger.info(f"AgentLoop initialized: mode={mode}, strategy={strategy}")
    
    async def run(self, query: str, max_iterations: int = 5) -> Dict[str, Any]:
        """
        Execute complete 5-phase pipeline
        
        Args:
            query: User query
            max_iterations: Max iterations for loop (safety limit)
            
        Returns:
            Final response with all phase outputs
        """
        logger.info(f"AgentLoop starting: query={query[:50]}...")
        
        try:
            # ===== PHASE 1: PERCEPTION =====
            perception_output = await self.perception.run(query, self.context_manager)
            if perception_output.get("status") == "error":
                return perception_output
            
            # ===== PHASE 2: DECISION =====
            decision_output = await self.decision.run(perception_output, self.context_manager)
            if decision_output.get("status") == "error":
                return decision_output
            
            # ===== PHASE 3: RETRIEVAL (Optional) =====
            route = perception_output.get("data", {}).get("route", "agentic")
            retrieval_output = None
            if route in ["retrieval", "hybrid"]:
                retrieval_output = await self.retrieval.run(query, self.context_manager)
            
            # ===== PHASE 4: EXECUTION =====
            execution_output = await self.executor.run(decision_output, self.context_manager)
            if execution_output.get("status") == "error":
                # Try fallback strategy
                execution_output = await self._execute_fallback(decision_output)
            
            # ===== PHASE 5: SUMMARIZATION =====
            summarization_output = await self.summarizer.run(execution_output, self.context_manager)
            
            # Build final response
            response = {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "mode": self.mode,
                "phases": {
                    "perception": perception_output,
                    "decision": decision_output,
                    "retrieval": retrieval_output,
                    "execution": execution_output,
                    "summarization": summarization_output
                },
                "session": self.context_manager.get_session_summary()
            }
            
            logger.info("AgentLoop completed successfully")
            return response
        
        except Exception as e:
            logger.error(f"AgentLoop error: {e}", exc_info=True)
            self.context_manager.record_error("agent_loop", str(e))
            return create_error_response(
                ComponentType.EXECUTION,
                "AGENT_LOOP_ERROR",
                f"Agent loop failed: {str(e)}",
                {"session": self.context_manager.get_session_summary()},
                recoverable=False
            )
    
    async def _execute_fallback(self, decision_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute fallback strategy when primary execution fails
        Implements self-correction by trying alternative paths
        """
        logger.info("Executing fallback strategy...")
        alternative_paths = decision_output.get("data", {}).get("alternative_paths", [])
        
        for path in alternative_paths[1:]:  # Skip primary path
            try:
                logger.info(f"Trying alternative path: {path.get('id')}")
                # Execute with fallback logic
                # For now, return success
                return await self.executor.run(decision_output, self.context_manager)
            except Exception as e:
                logger.warning(f"Fallback path failed: {e}")
                continue
        
        logger.error("All fallback paths exhausted")
        return create_error_response(
            ComponentType.EXECUTION,
            "EXECUTION_FAILED",
            "All execution strategies failed",
            recoverable=False
        )
