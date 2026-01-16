"""Decision Module - Graph-based planning and execution planning"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Set, List
from config.schemas import create_decision_output, create_error_response, ComponentType

logger = logging.getLogger(__name__)


class DecisionModule:
    """
    Phase 2: Create execution plan with graph-based planning
    
    Outputs:
    - Execution graph (nodes + edges)
    - Next step to execute
    - Alternative execution paths (A/B/C strategies)
    """
    
    def __init__(self, prompt_path: Optional[Path] = None):
        self.prompt_path = prompt_path
        self.prompt = self._load_prompt() if prompt_path else ""
        self.failed_tools: Set[str] = set()
    
    def _load_prompt(self) -> str:
        """Load decision prompt"""
        try:
            with open(self.prompt_path, 'r') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to load prompt: {e}")
            return ""
    
    async def run(self, perception_output: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """
        Generate execution plan based on perception output
        
        Args:
            perception_output: Output from perception module
            context: Context manager
            
        Returns:
            Decision output with execution graph
        """
        logger.info("Decision module generating execution plan...")
        
        try:
            # Extract perception data
            intent = perception_output.get("data", {}).get("interpreted_intent", "general")
            route = perception_output.get("data", {}).get("route", "agentic")
            
            # Build execution graph based on intent
            execution_graph = self._build_execution_graph(intent, route, context)
            
            # Determine next step
            next_step = self._determine_next_step(execution_graph, context)
            
            # Generate alternative paths (A/B/C strategies for self-correction)
            alternative_paths = self._generate_alternatives(intent, execution_graph)
            
            output = create_decision_output(
                execution_graph=execution_graph,
                next_step=next_step,
                alternative_paths=alternative_paths,
                reasoning=f"Plan based on {intent} analysis via {route} route",
                confidence=0.85
            )
            
            # Build graph in context
            context.build_graph_from_plan(execution_graph)
            context.add_global("decision_output", output)
            context.mark_step_completed("decision")
            
            logger.info(f"Decision plan created with {len(execution_graph.get('nodes', []))} steps")
            return output
        
        except Exception as e:
            logger.error(f"Decision error: {e}", exc_info=True)
            context.record_error("decision", str(e))
            return create_error_response(
                ComponentType.DECISION,
                "PLANNING_ERROR",
                f"Failed to create execution plan: {str(e)}",
                recoverable=True
            )
    
    def _build_execution_graph(self, intent: str, route: str, context: Any) -> Dict[str, Any]:
        """
        Build execution graph with nodes (tasks) and edges (dependencies)
        This is the GRAPH-BASED PLANNING implementation
        """
        nodes = []
        edges = []
        
        # Step 1: Retrieval (if needed)
        if route in ["retrieval", "hybrid"]:
            nodes.append({
                "id": "retrieval",
                "type": "retrieval",
                "description": "Retrieve relevant financial documents",
                "tool": "rag_pipeline"
            })
        
        # Step 2: Tool execution (main computation)
        if intent == "equity_valuation":
            tool = "calculate_dcf_valuation"
            description = "Calculate DCF valuation"
        elif intent == "financial_comparison":
            tool = "compare_financial_metrics"
            description = "Compare financial metrics"
        else:
            tool = "calculate_financial_metrics"
            description = "Analyze financial metrics"
        
        nodes.append({
            "id": "tool_execution",
            "type": "tool",
            "description": description,
            "tool": tool
        })
        
        # Step 3: Validation
        nodes.append({
            "id": "validation",
            "type": "code",
            "description": "Validate results",
            "tool": None
        })
        
        # Step 4: Summarization
        nodes.append({
            "id": "summarization",
            "type": "summarization",
            "description": "Generate summary",
            "tool": None
        })
        
        # Build edges (graph connections)
        if route in ["retrieval", "hybrid"]:
            edges.append({"from": "retrieval", "to": "tool_execution"})
        
        edges.append({"from": "tool_execution", "to": "validation"})
        edges.append({"from": "validation", "to": "summarization"})
        
        return {
            "nodes": nodes,
            "edges": edges
        }
    
    def _determine_next_step(self, execution_graph: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """Determine the next step to execute in the graph"""
        nodes = execution_graph.get("nodes", [])
        
        if not nodes:
            return {"id": "none", "action": "skip"}
        
        # Get first unexecuted node
        next_node = nodes[0]
        
        return {
            "id": next_node.get("id"),
            "action": "execute" if next_node.get("type") == "tool" else "process",
            "tool": next_node.get("tool"),
            "description": next_node.get("description")
        }
    
    def _generate_alternatives(self, intent: str, execution_graph: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate alternative execution paths (A/B/C strategies)
        Used for self-correction when primary path fails
        """
        alternatives = []
        
        # Path A: Direct execution (primary)
        alternatives.append({
            "id": "path_a",
            "name": "Direct Execution",
            "description": "Primary execution path",
            "order": 1
        })
        
        # Path B: With validation retry
        alternatives.append({
            "id": "path_b",
            "name": "With Validation Retry",
            "description": "Retry with stricter validation",
            "order": 2
        })
        
        # Path C: Simplified approach
        alternatives.append({
            "id": "path_c",
            "name": "Simplified Approach",
            "description": "Fallback to simplified calculation",
            "order": 3
        })
        
        return alternatives
