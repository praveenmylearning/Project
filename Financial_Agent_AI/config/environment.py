"""
Centralized Environment Configuration
All API keys and sensitive settings managed from one place
No need to scatter config across files
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from enum import Enum


class EnvironmentType(Enum):
    """Environment modes"""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


class APIProvider(Enum):
    """Available LLM Providers"""
    GOOGLE_GEMINI = "google"
    OPENAI = "openai"
    OLLAMA = "ollama"


# Load .env file from project root
_project_root = Path(__file__).parent.parent
_env_file = _project_root / ".env"
if _env_file.exists():
    load_dotenv(_env_file)


class EnvironmentConfig:
    """
    Centralized configuration manager
    Use this throughout your project instead of os.getenv scattered everywhere
    """
    
    # ==================== ENVIRONMENT ====================
    @property
    def environment(self) -> str:
        """Get current environment mode"""
        return os.getenv("ENVIRONMENT", "development")
    
    @property
    def is_production(self) -> bool:
        return self.environment == "production"
    
    @property
    def is_development(self) -> bool:
        return self.environment == "development"
    
    @property
    def is_testing(self) -> bool:
        return self.environment == "testing"
    
    # ==================== LLM PROVIDER ====================
    @property
    def llm_provider(self) -> str:
        """Get configured LLM provider: google, openai, or ollama"""
        return os.getenv("LLM_PROVIDER", "google").lower()
    
    @property
    def is_google_gemini(self) -> bool:
        return self.llm_provider == "google"
    
    @property
    def is_openai(self) -> bool:
        return self.llm_provider == "openai"
    
    @property
    def is_ollama(self) -> bool:
        return self.llm_provider == "ollama"
    
    # ==================== GOOGLE GEMINI ====================
    @property
    def google_api_key(self) -> Optional[str]:
        """Google Gemini API Key"""
        key = os.getenv("GOOGLE_API_KEY")
        if not key and self.is_google_gemini and self.is_production:
            raise ValueError("GOOGLE_API_KEY not set in environment variables")
        return key
    
    @property
    def google_model(self) -> str:
        """Google Gemini model to use"""
        return os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
    
    # ==================== OPENAI ====================
    @property
    def openai_api_key(self) -> Optional[str]:
        """OpenAI API Key"""
        key = os.getenv("OPENAI_API_KEY")
        if not key and self.is_openai and self.is_production:
            raise ValueError("OPENAI_API_KEY not set in environment variables")
        return key
    
    @property
    def openai_model(self) -> str:
        """OpenAI model to use"""
        return os.getenv("OPENAI_MODEL", "gpt-4-turbo")
    
    # ==================== OLLAMA ====================
    @property
    def ollama_url(self) -> str:
        """Ollama server URL"""
        return os.getenv("OLLAMA_URL", "http://localhost:11434")
    
    @property
    def ollama_model(self) -> str:
        """Ollama model to use"""
        return os.getenv("OLLAMA_MODEL", "mistral")
    
    # ==================== VECTOR DATABASE ====================
    @property
    def qdrant_url(self) -> str:
        """Qdrant vector database URL"""
        return os.getenv("QDRANT_URL", "http://localhost:6333")
    
    @property
    def qdrant_api_key(self) -> Optional[str]:
        """Qdrant API Key (optional)"""
        return os.getenv("QDRANT_API_KEY")
    
    @property
    def qdrant_collection(self) -> str:
        """Default Qdrant collection name"""
        return os.getenv("QDRANT_COLLECTION", "financial_documents")
    
    # ==================== MCP SERVERS ====================
    @property
    def mcp_finance_enabled(self) -> bool:
        """Enable finance tools MCP server"""
        return os.getenv("MCP_FINANCE_ENABLED", "true").lower() == "true"
    
    @property
    def mcp_web_enabled(self) -> bool:
        """Enable web tools MCP server"""
        return os.getenv("MCP_WEB_ENABLED", "true").lower() == "true"
    
    @property
    def mcp_document_enabled(self) -> bool:
        """Enable document tools MCP server"""
        return os.getenv("MCP_DOCUMENT_ENABLED", "true").lower() == "true"
    
    # ==================== DATA PATHS ====================
    @property
    def annual_reports_dir(self) -> Path:
        """Directory for annual reports"""
        data_dir = _project_root / "data" / "annual_reports"
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir
    
    @property
    def data_dir(self) -> Path:
        """Data directory"""
        data_dir = _project_root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir
    
    @property
    def logs_dir(self) -> Path:
        """Logs directory"""
        logs_dir = _project_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        return logs_dir
    
    # ==================== LOGGING ====================
    @property
    def log_level(self) -> str:
        """Logging level"""
        return os.getenv("LOG_LEVEL", "INFO")
    
    @property
    def enable_logging(self) -> bool:
        """Enable logging to files"""
        return os.getenv("ENABLE_LOGGING", "true").lower() == "true"
    
    # ==================== DEBUG & TESTING ====================
    @property
    def debug_mode(self) -> bool:
        """Enable debug mode"""
        return os.getenv("DEBUG", "false").lower() == "true"
    
    @property
    def verbose(self) -> bool:
        """Enable verbose output"""
        return os.getenv("VERBOSE", "false").lower() == "true"
    
    # ==================== SANDBOX & EXECUTION ====================
    @property
    def sandbox_enabled(self) -> bool:
        """Enable sandboxed execution for tools"""
        return os.getenv("SANDBOX_ENABLED", "true").lower() == "true"
    
    @property
    def max_tool_timeout(self) -> int:
        """Maximum timeout for tool execution (seconds)"""
        return int(os.getenv("MAX_TOOL_TIMEOUT", "30"))
    
    @property
    def max_retries(self) -> int:
        """Maximum retries for failed operations"""
        return int(os.getenv("MAX_RETRIES", "3"))
    
    # ==================== UTILITY METHODS ====================
    
    def get_all_settings(self) -> Dict[str, Any]:
        """Get all configuration settings as dictionary"""
        return {
            # Environment
            "environment": self.environment,
            "is_production": self.is_production,
            "is_development": self.is_development,
            
            # LLM Provider
            "llm_provider": self.llm_provider,
            "is_google_gemini": self.is_google_gemini,
            "is_openai": self.is_openai,
            "is_ollama": self.is_ollama,
            "google_model": self.google_model,
            "openai_model": self.openai_model,
            "ollama_model": self.ollama_model,
            "ollama_url": self.ollama_url,
            
            # Vector DB
            "qdrant_url": self.qdrant_url,
            "qdrant_collection": self.qdrant_collection,
            
            # MCP
            "mcp_finance_enabled": self.mcp_finance_enabled,
            "mcp_web_enabled": self.mcp_web_enabled,
            "mcp_document_enabled": self.mcp_document_enabled,
            
            # Execution
            "sandbox_enabled": self.sandbox_enabled,
            "max_tool_timeout": self.max_tool_timeout,
            "max_retries": self.max_retries,
            
            # Logging
            "log_level": self.log_level,
            "enable_logging": self.enable_logging,
            "debug_mode": self.debug_mode,
        }
    
    def validate_config(self) -> tuple[bool, list[str]]:
        """
        Validate configuration is correct for current environment
        Returns (is_valid, list_of_errors)
        """
        errors = []
        
        # Check LLM provider config
        if self.is_google_gemini and not self.google_api_key and self.is_production:
            errors.append("Google Gemini selected but GOOGLE_API_KEY not set")
        
        if self.is_openai and not self.openai_api_key and self.is_production:
            errors.append("OpenAI selected but OPENAI_API_KEY not set")
        
        # Check Qdrant if needed
        if not self.qdrant_url:
            errors.append("QDRANT_URL not configured")
        
        return len(errors) == 0, errors
    
    def print_config(self):
        """Print current configuration (safe - no secrets)"""
        print("\n" + "="*70)
        print("📋 ENVIRONMENT CONFIGURATION")
        print("="*70)
        
        settings = self.get_all_settings()
        
        # Group by category
        categories = {
            "Environment": ["environment", "is_production", "is_development"],
            "LLM Provider": ["llm_provider", "is_google_gemini", "is_openai", "is_ollama"],
            "Models": ["google_model", "openai_model", "ollama_model"],
            "Vector DB": ["qdrant_url", "qdrant_collection"],
            "MCP Servers": ["mcp_finance_enabled", "mcp_web_enabled", "mcp_document_enabled"],
            "Execution": ["sandbox_enabled", "max_tool_timeout", "max_retries"],
            "Logging": ["log_level", "enable_logging", "debug_mode"],
        }
        
        for category, keys in categories.items():
            print(f"\n{category}:")
            for key in keys:
                if key in settings:
                    print(f"  {key}: {settings[key]}")


# Global instance - use this everywhere
config = EnvironmentConfig()
