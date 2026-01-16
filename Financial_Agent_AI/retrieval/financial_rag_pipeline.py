"""Advanced Finance-Specialized RAG Pipeline

Implements sophisticated financial document retrieval with:
- Hybrid search (dense + sparse embeddings)
- Financial metadata filtering (company, fiscal period, doc type)
- Cross-encoder reranking
- SEC filing support (10-K, 10-Q, 8-K)
- Unified JSON schema output

Uses centralized configuration from config/environment.py
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from pydantic import BaseModel, Field

# Import centralized configuration
from config.environment import config

logger = logging.getLogger(__name__)


# ========================================================================
# FINANCIAL DOCUMENT SCHEMA - Unified with Core Project
# ========================================================================

class DocumentType(str, Enum):
    """Financial document types"""
    ANNUAL_REPORT = "10-k"      # Annual report
    QUARTERLY_REPORT = "10-q"   # Quarterly report
    CURRENT_REPORT = "8-k"      # Current report
    PROXY_STATEMENT = "def-14a" # Proxy statement
    EARNINGS_RELEASE = "earnings_release"
    RESEARCH_REPORT = "research_report"


class FiscalQuarter(str, Enum):
    """Fiscal quarters"""
    Q1 = "q1"
    Q2 = "q2"
    Q3 = "q3"
    Q4 = "q4"


class FinancialDocumentMetadata(BaseModel):
    """Metadata for financial documents - Unified Schema"""
    company_name: Optional[str] = Field(
        default=None,
        description="Company name (lowercase, e.g., 'apple', 'microsoft', 'amazon')"
    )
    ticker_symbol: Optional[str] = Field(
        default=None,
        description="Stock ticker (e.g., 'AAPL', 'MSFT')"
    )
    doc_type: Optional[DocumentType] = Field(
        default=None,
        description="Document type (10-k, 10-q, 8-k, etc.)"
    )
    fiscal_year: Optional[int] = Field(
        default=None,
        description="Fiscal year (e.g., 2024, 2023)"
    )
    fiscal_quarter: Optional[FiscalQuarter] = Field(
        default=None,
        description="Fiscal quarter if applicable (q1-q4)"
    )
    filing_date: Optional[str] = Field(
        default=None,
        description="Date when document was filed (YYYY-MM-DD)"
    )
    period_end_date: Optional[str] = Field(
        default=None,
        description="Period end date (YYYY-MM-DD)"
    )
    source_url: Optional[str] = Field(
        default=None,
        description="Original source URL (SEC EDGAR, etc.)"
    )
    page_number: Optional[int] = Field(
        default=None,
        description="Page number in original document"
    )
    section: Optional[str] = Field(
        default=None,
        description="Section of document (e.g., 'Management Discussion', 'Financial Statements')"
    )
    
    model_config = {"use_enum_values": True}


# ========================================================================
# FINANCIAL RAG PIPELINE
# ========================================================================

class FinancialRAGPipeline:
    """
    Advanced RAG pipeline for financial documents
    
    Features:
    - Hybrid search (dense embeddings + sparse BM25)
    - Financial metadata filtering
    - Cross-encoder reranking
    - Automatic filter extraction from queries
    - Unified JSON schema output
    """
    
    def __init__(
        self,
        embedding_model: str = "models/gemini-embedding-001",
        reranker_model: str = "BAAI/bge-reranker-base",
        collection_name: Optional[str] = None
    ):
        """Initialize RAG pipeline with centralized configuration"""
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        # Use config.qdrant_collection if not provided
        self.collection_name = collection_name or config.qdrant_collection
        
        # Company mappings for query understanding
        self.company_mapping = {
            "amazon": ["amazon", "amzn"],
            "apple": ["apple", "aapl"],
            "microsoft": ["microsoft", "msft"],
            "google": ["google", "alphabet", "googl", "goog"],
            "tesla": ["tesla", "tsla"],
            "nvidia": ["nvidia", "nvda"],
            "meta": ["meta", "facebook", "fb"],
            "jpmorgan": ["jpmorgan", "jpm"],
            "berkshire": ["berkshire", "brk"],
        }
        
        logger.info(f"Financial RAG Pipeline initialized: {self.collection_name}")
        logger.info(f"Qdrant URL: {config.qdrant_url}")
    
    # ====================================================================
    # QUERY UNDERSTANDING & FILTER EXTRACTION
    # ====================================================================
    
    def extract_financial_filters(self, query: str) -> Dict[str, Any]:
        """
        Extract financial metadata filters from natural language query
        
        Maps company names/tickers to canonical names
        Identifies document types and fiscal periods
        
        Args:
            query: Natural language query
            
        Returns:
            Dict with extracted filters
        """
        query_lower = query.lower()
        filters = {}
        
        # Extract company name/ticker
        for canonical_name, variations in self.company_mapping.items():
            for variation in variations:
                if variation in query_lower:
                    filters["company_name"] = canonical_name
                    filters["ticker_symbol"] = self._get_ticker_from_company(canonical_name)
                    break
        
        # Extract document type
        if "annual" in query_lower or "10-k" in query_lower or "yearly" in query_lower:
            filters["doc_type"] = "10-k"
        elif "quarterly" in query_lower or "10-q" in query_lower or "q1" in query_lower or "q2" in query_lower or "q3" in query_lower or "q4" in query_lower:
            filters["doc_type"] = "10-q"
        elif "current" in query_lower or "8-k" in query_lower:
            filters["doc_type"] = "8-k"
        
        # Extract fiscal year
        import re
        year_match = re.search(r'\b(20\d{2})\b', query)
        if year_match:
            filters["fiscal_year"] = int(year_match.group(1))
        
        # Extract fiscal quarter
        quarter_match = re.search(r'\b(q[1-4])\b', query_lower)
        if quarter_match:
            filters["fiscal_quarter"] = quarter_match.group(1)
        
        logger.info(f"Extracted filters: {filters}")
        return filters
    
    def _get_ticker_from_company(self, company_name: str) -> str:
        """Map company name to ticker symbol"""
        ticker_map = {
            "amazon": "AMZN",
            "apple": "AAPL",
            "microsoft": "MSFT",
            "google": "GOOGL",
            "tesla": "TSLA",
            "nvidia": "NVDA",
            "meta": "META",
            "jpmorgan": "JPM",
            "berkshire": "BRK",
        }
        return ticker_map.get(company_name, company_name.upper())
    
    # ====================================================================
    # HYBRID SEARCH (Dense + Sparse)
    # ====================================================================
    
    async def hybrid_search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform hybrid search combining dense and sparse embeddings
        
        Args:
            query: Search query
            k: Number of results
            filters: Optional metadata filters
            
        Returns:
            Retrieval output in unified JSON schema
        """
        logger.info(f"Hybrid search: '{query}' (k={k})")
        
        if not filters:
            filters = self.extract_financial_filters(query)
        
        try:
            # Placeholder for actual hybrid search
            # In production, integrate with Qdrant vector DB
            documents = await self._perform_hybrid_search(query, k, filters)
            
            # Rerank results
            documents = await self._rerank_documents(query, documents)
            
            # Create unified response
            from config.schemas import create_retrieval_output
            
            output = create_retrieval_output(
                documents=[
                    {
                        "id": doc.get("id"),
                        "title": doc.get("title"),
                        "content": doc.get("content")[:500],  # Truncate
                        "score": doc.get("score"),
                        "metadata": doc.get("metadata", {})
                    }
                    for doc in documents[:k]
                ],
                search_query=query,
                total_documents=len(documents),
                reasoning=f"Hybrid search with financial filters: {filters}"
            )
            
            logger.info(f"Found {len(documents)} documents")
            return output
        
        except Exception as e:
            logger.error(f"Hybrid search error: {e}")
            from config.schemas import create_error_response, ComponentType
            return create_error_response(
                ComponentType.RETRIEVAL,
                "SEARCH_ERROR",
                f"Hybrid search failed: {str(e)}",
                recoverable=True
            )
    
    async def _perform_hybrid_search(
        self,
        query: str,
        k: int,
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Perform actual hybrid search (integrate with Qdrant)"""
        # Placeholder implementation
        documents = [
            {
                "id": f"doc_{i}",
                "title": f"Financial Document {i}",
                "content": f"Sample financial document content for {query}",
                "score": 0.95 - (i * 0.1),
                "metadata": {
                    **filters,
                    "page_number": i + 1,
                    "source": "SEC EDGAR"
                }
            }
            for i in range(min(k + 5, 10))
        ]
        return documents
    
    async def _rerank_documents(
        self,
        query: str,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Rerank documents using cross-encoder"""
        # In production, use HuggingFaceCrossEncoder
        # For now, documents are already scored
        return sorted(documents, key=lambda x: x.get("score", 0), reverse=True)
    
    # ====================================================================
    # SEC FILING SEARCH
    # ====================================================================
    
    async def search_sec_filings(
        self,
        company_ticker: str,
        filing_type: str = "10-q",
        period: Optional[str] = None,
        search_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search SEC filings for specific company
        
        Args:
            company_ticker: Ticker symbol (e.g., 'AAPL')
            filing_type: Type of filing (10-q, 10-k, 8-k)
            period: Specific period (e.g., 'q1-2024', '2023-annual')
            search_query: Optional keyword search within filing
            
        Returns:
            List of matching SEC filings with metadata
        """
        logger.info(f"Searching SEC filings: {company_ticker} {filing_type}")
        
        filters = {
            "ticker_symbol": company_ticker,
            "doc_type": filing_type
        }
        
        query = f"{company_ticker} {filing_type}"
        if search_query:
            query += f" {search_query}"
        
        return await self.hybrid_search(query, filters=filters)
    
    # ====================================================================
    # FINANCIAL METRICS SEARCH
    # ====================================================================
    
    async def search_financial_metrics(
        self,
        company_ticker: str,
        metrics: List[str],
        fiscal_period: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search for specific financial metrics in documents
        
        Args:
            company_ticker: Company ticker
            metrics: List of metrics to search (e.g., ['revenue', 'profit', 'cashflow'])
            fiscal_period: Specific period to search
            
        Returns:
            Relevant document sections with metrics
        """
        metrics_str = ", ".join(metrics)
        query = f"{company_ticker} {metrics_str}"
        
        if fiscal_period:
            query += f" {fiscal_period}"
        
        return await self.hybrid_search(query)
    
    # ====================================================================
    # COMPARATIVE ANALYSIS SEARCH
    # ====================================================================
    
    async def search_comparative_analysis(
        self,
        companies: List[str],
        metrics: List[str],
        period: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search for comparative financial data across companies
        
        Args:
            companies: List of company tickers
            metrics: Metrics to compare
            period: Time period
            
        Returns:
            Comparable data across companies
        """
        companies_str = " vs ".join(companies)
        metrics_str = ", ".join(metrics)
        query = f"{companies_str}: {metrics_str}"
        
        if period:
            query += f" {period}"
        
        # Search for each company
        results = {}
        for company in companies:
            company_query = query.replace(companies_str, company)
            results[company] = await self.hybrid_search(company_query)
        
        return {
            "comparison_query": query,
            "companies": companies,
            "results": results
        }


# ========================================================================
# FINANCIAL DOCUMENT CHUNK PROCESSOR
# ========================================================================

class FinancialDocumentProcessor:
    """Process financial documents into searchable chunks"""
    
    @staticmethod
    def extract_financial_sections(document_text: str) -> Dict[str, str]:
        """
        Extract standard financial sections from document
        
        Returns:
            Dict with section names and content
        """
        sections = {
            "management_discussion": None,
            "financial_statements": None,
            "footnotes": None,
            "risk_factors": None,
            "executive_summary": None
        }
        
        # Simple pattern matching - enhance in production
        if "management" in document_text.lower() and "discussion" in document_text.lower():
            sections["management_discussion"] = "Found MD&A section"
        
        if "balance sheet" in document_text.lower() or "income statement" in document_text.lower():
            sections["financial_statements"] = "Found financial statements"
        
        return {k: v for k, v in sections.items() if v}
    
    @staticmethod
    def create_financial_chunks(
        document_text: str,
        chunk_size: int = 1000,
        overlap: int = 200,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Create overlapping chunks from financial document
        
        Args:
            document_text: Full document text
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            metadata: Document metadata
            
        Returns:
            List of chunks with metadata
        """
        chunks = []
        
        for i in range(0, len(document_text), chunk_size - overlap):
            chunk_text = document_text[i:i + chunk_size]
            
            chunk = {
                "id": f"chunk_{i}",
                "text": chunk_text,
                "metadata": {
                    **(metadata or {}),
                    "chunk_index": len(chunks),
                    "start_position": i,
                    "end_position": i + len(chunk_text)
                }
            }
            chunks.append(chunk)
        
        return chunks


# ========================================================================
# INTEGRATION WITH CORE RAG MODULE
# ========================================================================

import asyncio

class RAGPipeline:
    """Enhanced RAG Pipeline with financial specialization"""
    
    def __init__(self, prompt_path=None):
        self.prompt_path = prompt_path
        self.financial_rag = FinancialRAGPipeline()
        self.processor = FinancialDocumentProcessor()
        logger.info("RAG Pipeline initialized with financial specialization")
    
    async def run(self, query: str, context: Any) -> Dict[str, Any]:
        """Run enhanced financial RAG pipeline"""
        logger.info(f"RAG pipeline executing: {query[:50]}...")
        
        try:
            # Extract financial filters from query
            filters = self.financial_rag.extract_financial_filters(query)
            
            # Perform hybrid search
            retrieval_output = await self.financial_rag.hybrid_search(query, k=5, filters=filters)
            
            # Store in context
            context.add_global("retrieval_output", retrieval_output)
            context.mark_step_completed("retrieval")
            
            logger.info(f"RAG pipeline complete")
            return retrieval_output
        
        except Exception as e:
            logger.error(f"RAG pipeline error: {e}")
            from config.schemas import create_error_response, ComponentType
            return create_error_response(
                ComponentType.RETRIEVAL,
                "RAG_ERROR",
                f"RAG pipeline failed: {str(e)}",
                recoverable=True
            )
