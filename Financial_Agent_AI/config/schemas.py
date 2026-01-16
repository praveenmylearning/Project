"""
Unified JSON Schema for all components - Standardized Communication
Ensures consistent data structure across Perception → Decision → Retrieval → Execution → Summarization

Uses Pydantic v2 for robust type validation and serialization.
"""

from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ConfigDict


class ComponentType(str, Enum):
    """Component types in the agent pipeline"""
    PERCEPTION = "perception"
    DECISION = "decision"
    RETRIEVAL = "retrieval"
    EXECUTION = "execution"
    SUMMARIZATION = "summarization"


class StatusType(str, Enum):
    """Status types for operations"""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"
    PENDING = "pending"


# ============================================================================
# PERCEPTION MODULE SCHEMA
# ============================================================================

class PerceptionData(BaseModel):
    """Perception phase data"""
    interpreted_intent: str = Field(..., description="Intent extracted from user query")
    route: str = Field(..., description="Routing decision: agentic, retrieval, or hybrid")
    entities: List[str] = Field(default_factory=list, description="Extracted entities")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")
    reasoning: str = Field(default="", description="Explanation of routing decision")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator('route')
    @classmethod
    def validate_route(cls, v):
        if v not in ["agentic", "retrieval", "hybrid"]:
            raise ValueError("route must be 'agentic', 'retrieval', or 'hybrid'")
        return v

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "interpreted_intent": "equity_valuation",
            "route": "hybrid",
            "entities": ["AAPL", "Apple"],
            "confidence": 0.95,
            "reasoning": "Query mentions company and financial metrics",
            "metadata": {"user_id": "user_123"}
        }
    })


class PerceptionOutput(BaseModel):
    """Complete Perception phase output"""
    component: ComponentType = ComponentType.PERCEPTION
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    status: StatusType = StatusType.SUCCESS
    data: PerceptionData

    def dict(self, **kwargs):
        return super().model_dump(**kwargs)


# ============================================================================
# DECISION MODULE SCHEMA
# ============================================================================

class ExecutionNode(BaseModel):
    """Node in execution graph"""
    node_id: str = Field(..., description="Unique node identifier")
    task: str = Field(..., description="Task description")
    tool: Optional[str] = Field(None, description="Tool to use")
    dependencies: List[str] = Field(default_factory=list, description="IDs of dependent nodes")
    priority: int = Field(default=1, ge=1, description="Execution priority")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "node_id": "node_1",
            "task": "Calculate DCF valuation",
            "tool": "calculate_advanced_intrinsic_value",
            "dependencies": [],
            "priority": 1
        }
    })


class ExecutionGraph(BaseModel):
    """Directed acyclic graph of tasks"""
    nodes: List[ExecutionNode] = Field(..., description="Graph nodes")
    edges: List[Tuple[str, str]] = Field(..., description="Graph edges (from, to)")
    start_node: str = Field(..., description="Starting node ID")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "nodes": [{"node_id": "1", "task": "Extract metrics"}],
            "edges": [("1", "2")],
            "start_node": "1"
        }
    })


class DecisionData(BaseModel):
    """Decision phase data"""
    execution_graph: ExecutionGraph = Field(..., description="Task execution graph")
    next_step: Dict[str, Any] = Field(..., description="Next immediate step")
    alternative_paths: List[Dict[str, Any]] = Field(default_factory=list, description="Alternative execution paths A→B→C")
    reasoning: str = Field(default="", description="Decision reasoning")
    confidence: float = Field(default=0.85, ge=0.0, le=1.0, description="Confidence score")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "next_step": {"action": "retrieve_financial_data"},
            "alternative_paths": [{"path": "A"}, {"path": "B"}],
            "confidence": 0.90
        }
    })


class DecisionOutput(BaseModel):
    """Complete Decision phase output"""
    component: ComponentType = ComponentType.DECISION
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    status: StatusType = StatusType.SUCCESS
    data: DecisionData

    def dict(self, **kwargs):
        return super().model_dump(**kwargs)


# ============================================================================
# RETRIEVAL MODULE SCHEMA
# ============================================================================

class FinancialDocument(BaseModel):
    """Financial document with metadata"""
    doc_id: str = Field(..., description="Document unique ID")
    title: str = Field(..., description="Document title")
    company_name: str = Field(..., description="Company name")
    ticker_symbol: Optional[str] = Field(None, description="Stock ticker symbol")
    doc_type: str = Field(..., description="10-k, 10-q, 8-k, etc.")
    fiscal_year: int = Field(..., description="Fiscal year")
    fiscal_quarter: Optional[str] = Field(None, description="q1, q2, q3, q4")
    filing_date: str = Field(..., description="Filing date YYYY-MM-DD")
    section: Optional[str] = Field(None, description="Section reference")
    content_chunk: str = Field(..., description="Relevant text chunk")
    relevance_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Relevance score")
    source_url: Optional[str] = Field(None, description="Source URL")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "doc_id": "AAPL_10K_2024_001",
            "title": "Apple Inc. 10-K 2024",
            "company_name": "apple",
            "ticker_symbol": "AAPL",
            "doc_type": "10-k",
            "fiscal_year": 2024,
            "fiscal_quarter": None,
            "filing_date": "2024-01-10",
            "section": "Item 7 - Financial Statements",
            "content_chunk": "Total revenue for FY2024...",
            "relevance_score": 0.95
        }
    })


class RetrievalData(BaseModel):
    """Retrieval phase data"""
    search_query: str = Field(..., description="Original search query")
    documents: List[FinancialDocument] = Field(default_factory=list, description="Retrieved documents")
    total_documents: int = Field(..., description="Total documents found")
    search_filters: Dict[str, Any] = Field(default_factory=dict, description="Applied filters")
    reasoning: str = Field(default="", description="Search strategy explanation")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "search_query": "Apple Q1 2024 revenue",
            "total_documents": 42,
            "search_filters": {"company": "apple", "fiscal_period": "q1"}
        }
    })


class RetrievalOutput(BaseModel):
    """Complete Retrieval phase output"""
    component: ComponentType = ComponentType.RETRIEVAL
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    status: StatusType = StatusType.SUCCESS
    data: RetrievalData

    def dict(self, **kwargs):
        return super().model_dump(**kwargs)


# ============================================================================
# EXECUTION MODULE SCHEMA
# ============================================================================

class ExecutionData(BaseModel):
    """Execution phase data"""
    results: Dict[str, Any] = Field(..., description="Tool execution results")
    tool_used: str = Field(..., description="Tool name executed")
    execution_time: float = Field(..., ge=0, description="Execution time in seconds")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")
    confidence: float = Field(default=0.9, ge=0.0, le=1.0, description="Result confidence")
    fallback_used: bool = Field(default=False, description="Whether fallback strategy was used")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "results": {"intrinsic_value": 150.45, "valuation_range": [140, 160]},
            "tool_used": "calculate_advanced_intrinsic_value",
            "execution_time": 0.234,
            "errors": [],
            "confidence": 0.92
        }
    })


class ExecutionOutput(BaseModel):
    """Complete Execution phase output"""
    component: ComponentType = ComponentType.EXECUTION
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    status: StatusType = StatusType.SUCCESS
    data: ExecutionData

    def dict(self, **kwargs):
        return super().model_dump(**kwargs)


# ============================================================================
# SUMMARIZATION MODULE SCHEMA
# ============================================================================

class Citation(BaseModel):
    """Citation reference"""
    source: str = Field(..., description="Source title or URL")
    page: Optional[int] = Field(None, description="Page number if applicable")
    date: Optional[str] = Field(None, description="Publication date")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "source": "Apple 10-K 2024",
            "page": 42,
            "date": "2024-01-10"
        }
    })


class SummarizationData(BaseModel):
    """Summarization phase data"""
    summary: str = Field(..., description="Executive summary")
    key_insights: List[str] = Field(default_factory=list, description="Key findings")
    citations: List[Citation] = Field(default_factory=list, description="Source citations")
    confidence: float = Field(default=0.85, ge=0.0, le=1.0, description="Summary confidence")
    next_questions: Optional[List[str]] = Field(None, description="Suggested follow-up questions")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "summary": "Apple's revenue grew 15% YoY...",
            "key_insights": ["Strong margin expansion", "Services growth"],
            "citations": [{"source": "Apple 10-K 2024", "page": 42}],
            "confidence": 0.88
        }
    })


class SummarizationOutput(BaseModel):
    """Complete Summarization phase output"""
    component: ComponentType = ComponentType.SUMMARIZATION
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    status: StatusType = StatusType.SUCCESS
    data: SummarizationData

    def dict(self, **kwargs):
        return super().model_dump(**kwargs)


# ============================================================================
# ERROR SCHEMA
# ============================================================================

class ErrorDetail(BaseModel):
    """Error detail information"""
    type: str = Field(..., description="Error type (validation, execution, etc.)")
    message: str = Field(..., description="Error message")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional error details")
    recoverable: bool = Field(default=False, description="Whether error is recoverable")
    suggested_recovery: Optional[str] = Field(None, description="Suggested recovery action")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "type": "validation_error",
            "message": "Invalid company ticker",
            "details": {"provided": "ABC", "expected": "AAPL"},
            "recoverable": True,
            "suggested_recovery": "Use company name instead"
        }
    })


class ErrorResponse(BaseModel):
    """Complete error response"""
    component: ComponentType = Field(..., description="Component where error occurred")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    status: StatusType = StatusType.ERROR
    error: ErrorDetail

    def dict(self, **kwargs):
        return super().model_dump(**kwargs)


# ============================================================================
# BACKWARD COMPATIBILITY HELPER FUNCTIONS
# ============================================================================

def create_perception_output(
    interpreted_intent: str,
    route: str,
    entities: List[str],
    confidence: float,
    reasoning: str = "",
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create perception output using pydantic model"""
    data = PerceptionData(
        interpreted_intent=interpreted_intent,
        route=route,
        entities=entities,
        confidence=confidence,
        reasoning=reasoning,
        metadata=metadata or {}
    )
    output = PerceptionOutput(data=data)
    return output.model_dump()


def create_decision_output(
    execution_graph: Dict[str, Any],
    next_step: Dict[str, Any],
    alternative_paths: List[Dict[str, Any]],
    reasoning: str = "",
    confidence: float = 0.85
) -> Dict[str, Any]:
    """Create decision output using pydantic model"""
    graph = ExecutionGraph(
        nodes=[ExecutionNode(**n) if isinstance(n, dict) else n for n in execution_graph.get("nodes", [])],
        edges=execution_graph.get("edges", []),
        start_node=execution_graph.get("start_node", "")
    )
    data = DecisionData(
        execution_graph=graph,
        next_step=next_step,
        alternative_paths=alternative_paths,
        reasoning=reasoning,
        confidence=confidence
    )
    output = DecisionOutput(data=data)
    return output.model_dump()


def create_retrieval_output(
    documents: List[Dict[str, Any]],
    search_query: str,
    total_documents: int,
    reasoning: str = "",
    search_filters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create retrieval output using pydantic model"""
    doc_objects = [FinancialDocument(**d) if isinstance(d, dict) else d for d in documents]
    data = RetrievalData(
        search_query=search_query,
        documents=doc_objects,
        total_documents=total_documents,
        reasoning=reasoning,
        search_filters=search_filters or {}
    )
    output = RetrievalOutput(data=data)
    return output.model_dump()


def create_execution_output(
    results: Dict[str, Any],
    tool_used: str,
    execution_time: float,
    errors: Optional[List[str]] = None,
    confidence: float = 0.9,
    fallback_used: bool = False
) -> Dict[str, Any]:
    """Create execution output using pydantic model"""
    data = ExecutionData(
        results=results,
        tool_used=tool_used,
        execution_time=execution_time,
        errors=errors or [],
        confidence=confidence,
        fallback_used=fallback_used
    )
    status = StatusType.ERROR if errors else StatusType.SUCCESS
    output = ExecutionOutput(data=data, status=status)
    return output.model_dump()


def create_summarizer_output(
    summary: str,
    key_insights: List[str],
    citations: Optional[List[Dict[str, str]]] = None,
    confidence: float = 0.85,
    next_questions: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Create summarization output using pydantic model"""
    citation_objects = [Citation(**c) if isinstance(c, dict) else c for c in (citations or [])]
    data = SummarizationData(
        summary=summary,
        key_insights=key_insights,
        citations=citation_objects,
        confidence=confidence,
        next_questions=next_questions
    )
    output = SummarizationOutput(data=data)
    return output.model_dump()


def create_error_response(
    component: ComponentType,
    error_type: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    recoverable: bool = False,
    suggested_recovery: Optional[str] = None
) -> Dict[str, Any]:
    """Create error response using pydantic model"""
    error = ErrorDetail(
        type=error_type,
        message=message,
        details=details or {},
        recoverable=recoverable,
        suggested_recovery=suggested_recovery
    )
    output = ErrorResponse(component=component, error=error)
    return output.model_dump()


# ============================================================================
# VALIDATION
# ============================================================================

def validate_component_output(output: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate component output against schema"""
    errors = []
    
    required_keys = ["component", "timestamp", "status"]
    for key in required_keys:
        if key not in output:
            errors.append(f"Missing required key: {key}")
    
    if "status" in output:
        if output["status"] not in [s.value for s in StatusType]:
            errors.append(f"Invalid status: {output['status']}")
    
    if "component" in output:
        if output["component"] not in [c.value for c in ComponentType]:
            errors.append(f"Invalid component: {output['component']}")
    
    return len(errors) == 0, errors

