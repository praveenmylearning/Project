"""Context Manager - Maintains state across all modules
Provides centralized access to configuration, state, and shared data
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

# Import centralized config
from config.environment import config

logger = logging.getLogger(__name__)


class ContextManager:
    """
    Manages context across all agent modules:
    - Stores query and results
    - Tracks execution flow
    - Manages state between modules
    - Provides centralized config access
    - Tracks failures and retry logic
    """
    
    def __init__(self, user_id: str = "default"):
        """Initialize context manager"""
        self.user_id = user_id
        self.query: str = ""
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
        # Module outputs
        self.perception_output: Optional[Dict[str, Any]] = None
        self.decision_output: Optional[Dict[str, Any]] = None
        self.retrieval_output: Optional[Dict[str, Any]] = None
        self.execution_output: Optional[Dict[str, Any]] = None
        self.summarizer_output: Optional[Dict[str, Any]] = None
        
        # Execution tracking
        self.execution_history: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, str]] = []
        self.failed_tools: set = set()
        self.successful_tools: set = set()
        
        # Shared data
        self.shared_data: Dict[str, Any] = {}
        
        logger.info(f"ContextManager initialized for user: {user_id}")

        self.execution_graph: Dict[str, Any] = {}
        self.completed_steps: Set[str] = set()
        self.failed_tools: Set[str] = set()
        self.tool_retry_counts: Dict[str, int] = {}
        self.session_start = datetime.now()
    
    def add_global(self, key: str, value: Any) -> None:
        """Add to global state"""
        self.global_state[key] = value
    
    def get_global(self, key: str, default: Any = None) -> Any:
        """Retrieve from global state"""
        return self.global_state.get(key, default)
    
    def add_step_result(self, step: str, result: Any) -> None:
        """Store result from a specific step"""
        self.step_results[step] = result
    
    def get_step_result(self, step: str, default: Any = None) -> Any:
        """Retrieve result from specific step"""
        return self.step_results.get(step, default)
    
    def record_error(self, component: str, error: str) -> None:
        """Record error for recovery and adaptation"""
        self.errors.append({
            "component": component,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })
        logger.warning(f"Error recorded in {component}: {error}")
    
    def build_graph_from_plan(self, execution_graph: Dict[str, Any]) -> None:
        """Store execution graph from decision module"""
        self.execution_graph = execution_graph
    
    def mark_step_completed(self, step: str) -> None:
        """Mark a step as completed"""
        self.completed_steps.add(step)
    
    def mark_step_failed(self, step: str) -> None:
        """Mark a step as failed"""
        self.failed_tools.add(step)
    
    def is_step_completed(self, step: str) -> bool:
        """Check if step was completed"""
        return step in self.completed_steps
    
    def get_failed_tools(self) -> Set[str]:
        """Get all failed tools (for self-correction)"""
        return self.failed_tools
    
    def record_tool_retry(self, tool: str) -> None:
        """Track retries per tool"""
        self.tool_retry_counts[tool] = self.tool_retry_counts.get(tool, 0) + 1
    
    def should_retry_tool(self, tool: str, max_retries: int = 3) -> bool:
        """Check if tool should be retried"""
        return self.tool_retry_counts.get(tool, 0) < max_retries
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of session for debugging/logging"""
        return {
            "duration": str(datetime.now() - self.session_start),
            "completed_steps": list(self.completed_steps),
            "failed_steps": list(self.failed_tools),
            "total_errors": len(self.errors),
            "tool_retries": self.tool_retry_counts,
            "errors": self.errors
        }
