"""Retrieval Module - Advanced Finance-Specialized RAG Pipeline

Implements sophisticated financial document retrieval with:
- Hybrid search (dense + sparse embeddings)
- Financial metadata filtering (company, fiscal period, doc type)
- SEC filing support (10-K, 10-Q, 8-K)
- Cross-encoder reranking
- Unified JSON schema output
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
from config.schemas import create_retrieval_output, create_error_response, ComponentType
from retrieval.financial_rag_pipeline import FinancialRAGPipeline

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Phase 3: Advanced Financial RAG Pipeline
    
    Features:
    - Hybrid search (vector + BM25)
    - Financial metadata extraction & filtering
    - SEC filing search (10-K, 10-Q, 8-K)
    - Financial metrics search
    - Comparative analysis
    - Cross-encoder reranking
    - Unified JSON schema output
    """
    
    def __init__(self, prompt_path: Optional[Path] = None):
        self.prompt_path = prompt_path
        self.prompt = self._load_prompt() if prompt_path else ""
        self.financial_rag = FinancialRAGPipeline()
    
    def _load_prompt(self) -> str:
        """Load retrieval prompt"""
        try:
            with open(self.prompt_path, 'r') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to load prompt: {e}")
            return ""
    
    async def run(self, query: str, context: Any) -> Dict[str, Any]:
        """
        Execute advanced financial RAG pipeline
        
        Args:
            query: Search query
            context: Context manager
            
        Returns:
            Retrieval output with financial documents
        """
        logger.info(f"Financial RAG Pipeline starting: {query[:50]}...")
        
        try:
            # Extract financial filters from query
            filters = self.financial_rag.extract_financial_filters(query)
            logger.info(f"Extracted financial filters: {filters}")
            
            # Perform hybrid search
            retrieval_output = await self.financial_rag.hybrid_search(query, k=5, filters=filters)
            
            # Store in context
            context.add_global("retrieval_output", retrieval_output)
            context.mark_step_completed("retrieval")
            
            # Get document count
            doc_count = len(retrieval_output.get("data", {}).get("documents", []))
            logger.info(f"Financial RAG complete: {doc_count} documents retrieved")
            
            return retrieval_output
        
        except Exception as e:
            logger.error(f"Financial RAG error: {e}", exc_info=True)
            context.record_error("retrieval", str(e))
            return create_error_response(
                ComponentType.RETRIEVAL,
                "RAG_ERROR",
                f"Financial RAG pipeline failed: {str(e)}",
                recoverable=True
            )
    
    # ====================================================================
    # SPECIALIZED SEARCH METHODS
    # ====================================================================
    
    async def search_sec_filings(
        self,
        company_ticker: str,
        filing_type: str = "10-q",
        query: Optional[str] = None,
        context: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Search specific SEC filings"""
        logger.info(f"SEC filing search: {company_ticker} {filing_type}")
        
        try:
            result = await self.financial_rag.search_sec_filings(
                company_ticker, filing_type, search_query=query
            )
            
            if context:
                context.add_global("sec_filing_results", result)
            
            return result
        
        except Exception as e:
            logger.error(f"SEC filing search error: {e}")
            return create_error_response(
                ComponentType.RETRIEVAL,
                "SEC_SEARCH_ERROR",
                f"SEC filing search failed: {str(e)}",
                recoverable=True
            )
    
    async def search_financial_metrics(
        self,
        company_ticker: str,
        metrics: list,
        fiscal_period: Optional[str] = None,
        context: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Search for specific financial metrics"""
        logger.info(f"Financial metrics search: {company_ticker} - {metrics}")
        
        try:
            result = await self.financial_rag.search_financial_metrics(
                company_ticker, metrics, fiscal_period
            )
            
            if context:
                context.add_global("metrics_results", result)
            
            return result
        
        except Exception as e:
            logger.error(f"Metrics search error: {e}")
            return create_error_response(
                ComponentType.RETRIEVAL,
                "METRICS_SEARCH_ERROR",
                f"Financial metrics search failed: {str(e)}",
                recoverable=True
            )
    
    async def comparative_analysis(
        self,
        companies: list,
        metrics: list,
        period: Optional[str] = None,
        context: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Search for comparative financial data"""
        logger.info(f"Comparative analysis: {companies} - {metrics}")
        
        try:
            result = await self.financial_rag.search_comparative_analysis(
                companies, metrics, period
            )
            
            if context:
                context.add_global("comparative_results", result)
            
            return result
        
        except Exception as e:
            logger.error(f"Comparative analysis error: {e}")
            return create_error_response(
                ComponentType.RETRIEVAL,
                "ANALYSIS_ERROR",
                f"Comparative analysis failed: {str(e)}",
                recoverable=True
            )

