"""
LLM Provider Configuration - Uses Centralized Environment Config
Supports: Google Gemini, OpenAI GPT-4, Ollama (local)

All API keys and settings come from config/environment.py
No need to manage environment variables in multiple places
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

# Import centralized environment config
try:
    from config.environment import config
except ImportError:
    from environment import config

logger = logging.getLogger(__name__)


# ===================== ABSTRACT PROVIDER =====================

class LLMProvider(ABC):
    """Base class for LLM providers"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from LLM"""
        pass
    
    def is_configured(self) -> bool:
        """Check if provider is configured"""
        raise NotImplementedError
    
    def test_connection(self) -> bool:
        """Test connection to provider"""
        raise NotImplementedError


# ===================== CONFIGURATION CLASS =====================
    """LLM Configuration management"""
    
    # Supported providers
    PROVIDERS = {
        "google": {
            "name": "Google Gemini",
            "env_key": "GOOGLE_API_KEY",
            "required_packages": ["google-genai"],
            "setup_url": "https://ai.google.dev/tutorials/python_quickstart",
            "free_tier": True,
            "models": ["gemini-2.0-flash", "gemini-1.5-pro"]
        },
        "openai": {
            "name": "OpenAI GPT-4",
            "env_key": "OPENAI_API_KEY",
            "required_packages": ["openai"],
            "setup_url": "https://platform.openai.com/account/api-keys",
            "free_tier": False,
            "models": ["gpt-4-turbo", "gpt-4-turbo-preview"]
        },
        "ollama": {
            "name": "Ollama (Local)",
            "env_key": "OLLAMA_URL",
            "required_packages": ["ollama"],
            "setup_url": "https://ollama.ai",
            "free_tier": True,
            "models": ["llama2", "mistral", "neural-chat"],
            "default_url": "http://localhost:11434"
        }
    }
    
    @staticmethod
    def check_provider_status() -> Dict[str, Any]:
        """Check status of all LLM providers"""
        status = {}
        
        for provider_key, provider_info in LLMConfig.PROVIDERS.items():
            env_key = provider_info["env_key"]
            is_configured = bool(os.getenv(env_key))
            
            status[provider_key] = {
                "name": provider_info["name"],
                "configured": is_configured,
                "env_variable": env_key,
                "free_tier": provider_info["free_tier"]
            }
        
        return status
    
    @staticmethod
    def get_setup_instructions(provider: str) -> str:
        """Get setup instructions for specific provider"""
        
        if provider == "google":
            return """
╔════════════════════════════════════════════════════════════════╗
║  GOOGLE GEMINI SETUP (RECOMMENDED - Free & Easy)              ║
╚════════════════════════════════════════════════════════════════╝

STEP 1: Get API Key
  1. Go to https://ai.google.dev/tutorials/python_quickstart
  2. Click "Get API Key" button
  3. Create new API key (free tier available)
  4. Copy the API key

STEP 2: Set Environment Variable
  
  Windows (PowerShell):
  ─────────────────────
  $env:GOOGLE_API_KEY = "your-api-key-here"
  
  Linux/Mac (Bash):
  ────────────────
  export GOOGLE_API_KEY="your-api-key-here"
  
  Or in .env file:
  ───────────────
  GOOGLE_API_KEY=your-api-key-here

STEP 3: Test Configuration
  python -c "import os; print('✅ GOOGLE_API_KEY configured' if os.getenv('GOOGLE_API_KEY') else '❌ Not configured')"

STEP 4: Use in Code
  from config.llm_config import get_llm_provider
  
  provider = get_llm_provider("google")
  response = provider.generate("Your prompt here")

MODELS AVAILABLE:
  - gemini-2.0-flash (Fastest, recommended)
  - gemini-1.5-pro (Most capable)

FREE TIER LIMITS:
  - 60 requests/minute
  - Perfect for development & testing
            """
        
        elif provider == "openai":
            return """
╔════════════════════════════════════════════════════════════════╗
║  OPENAI GPT-4 SETUP (Most Reliable)                           ║
╚════════════════════════════════════════════════════════════════╝

STEP 1: Create OpenAI Account
  1. Go to https://platform.openai.com/account/api-keys
  2. Sign up or log in
  3. Create new API key
  4. Copy the key (save securely)

STEP 2: Set Environment Variable
  
  Windows (PowerShell):
  ─────────────────────
  $env:OPENAI_API_KEY = "your-api-key-here"
  
  Linux/Mac (Bash):
  ────────────────
  export OPENAI_API_KEY="your-api-key-here"
  
  Or in .env file:
  ───────────────
  OPENAI_API_KEY=your-api-key-here

STEP 3: Add Payment Method
  - Go to Billing settings
  - Add credit card for usage
  - Set usage limits to prevent overspend

STEP 4: Test Configuration
  python -c "import os; print('✅ OPENAI_API_KEY configured' if os.getenv('OPENAI_API_KEY') else '❌ Not configured')"

STEP 5: Use in Code
  from config.llm_config import get_llm_provider
  
  provider = get_llm_provider("openai")
  response = provider.generate("Your prompt here")

MODELS AVAILABLE:
  - gpt-4-turbo (Fastest, recommended)
  - gpt-4-turbo-preview
  - gpt-4 (Most capable but slower)

PRICING:
  - Input: $0.01/1K tokens
  - Output: $0.03/1K tokens
            """
        
        elif provider == "ollama":
            return """
╔════════════════════════════════════════════════════════════════╗
║  OLLAMA SETUP (Local & Free)                                  ║
╚════════════════════════════════════════════════════════════════╝

STEP 1: Install Ollama
  1. Go to https://ollama.ai
  2. Download for your OS (macOS, Linux, Windows)
  3. Run installer
  4. Open terminal/PowerShell

STEP 2: Download and Run Model
  
  Basic:
  ──────
  ollama pull mistral
  ollama serve
  
  Or with specific model:
  ──────────────────────
  ollama pull llama2
  ollama serve

STEP 3: Set Environment Variable
  
  Windows (PowerShell):
  ─────────────────────
  $env:OLLAMA_URL = "http://localhost:11434"
  
  Linux/Mac (Bash):
  ────────────────
  export OLLAMA_URL="http://localhost:11434"
  
  Or in .env file:
  ───────────────
  OLLAMA_URL=http://localhost:11434

STEP 4: Test Configuration
  python -c "import os; print('✅ OLLAMA_URL configured' if os.getenv('OLLAMA_URL') else '❌ Not configured')"

STEP 5: Use in Code
  from config.llm_config import get_llm_provider
  
  provider = get_llm_provider("ollama")
  response = provider.generate("Your prompt here")

AVAILABLE MODELS:
  - mistral (Recommended - fast & capable)
  - llama2 (Good for financial analysis)
  - neural-chat (Optimized for conversation)
  - dolphin-mixtral (Advanced reasoning)

REQUIREMENTS:
  - 8GB+ RAM recommended
  - 10-20GB disk space for models
  - Model stays running in background
            """
        
        return "Unknown provider"
    
    @staticmethod
    def test_llm_connection(provider: str) -> bool:
        """Test LLM connection"""
        try:
            if provider == "google":
                import google.genai as genai
                api_key = os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    logger.error("GOOGLE_API_KEY not configured")
                    return False
                genai.configure(api_key=api_key)
                logger.info("✅ Google Gemini connection successful")
                return True
            
            elif provider == "openai":
                from openai import OpenAI
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    logger.error("OPENAI_API_KEY not configured")
                    return False
                client = OpenAI(api_key=api_key)
                logger.info("✅ OpenAI connection successful")
                return True
            
            elif provider == "ollama":
                import requests
                url = os.getenv("OLLAMA_URL", "http://localhost:11434")
                response = requests.get(f"{url}/api/tags")
                if response.status_code == 200:
                    logger.info("✅ Ollama connection successful")
                    return True
                else:
                    logger.error("Ollama not running or invalid URL")
                    return False
        
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


class LLMProvider:
    """Base LLM provider interface"""
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from LLM"""
        raise NotImplementedError


class GoogleGeminiProvider(LLMProvider):
    """Google Gemini provider"""
    
    def __init__(self, model: Optional[str] = None):
        try:
            import google.generativeai as genai
            
            api_key = config.google_api_key
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not set in environment")
            
            genai.configure(api_key=api_key)
            
            model_name = model or config.google_model
            self.model = genai.GenerativeModel(model_name)
            self.model_name = model_name
            
            logger.info(f"✅ Google Gemini initialized with model: {model_name}")
        except ImportError:
            raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")
        except Exception as e:
            logger.error(f"Failed to initialize Google Gemini: {e}")
            raise
    
    def generate(self, prompt: str, temperature: float = 0.7, **kwargs) -> str:
        """Generate using Google Gemini"""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"temperature": temperature}
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise


class OpenAIProvider(LLMProvider):
    """OpenAI provider"""
    
    def __init__(self, model: Optional[str] = None):
        try:
            from openai import OpenAI
            
            api_key = config.openai_api_key
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set in environment")
            
            self.client = OpenAI(api_key=api_key)
            
            model_name = model or config.openai_model
            self.model_name = model_name
            
            logger.info(f"✅ OpenAI initialized with model: {model_name}")
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
            raise
    
    def generate(self, prompt: str, temperature: float = 0.7, **kwargs) -> str:
        """Generate using OpenAI"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise


class OllamaProvider(LLMProvider):
    """Ollama provider"""
    
    def __init__(self, model: Optional[str] = None, url: Optional[str] = None):
        try:
            import requests
            
            self.url = url or config.ollama_url
            self.model_name = model or config.ollama_model
            self.session = requests.Session()
            
            # Test connection
            response = self.session.get(f"{self.url}/api/tags", timeout=5)
            response.raise_for_status()
            
            logger.info(f"✅ Ollama initialized at {self.url} with model: {self.model_name}")
        except ImportError:
            raise ImportError("requests package not installed. Run: pip install requests")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama: {e}")
            raise ConnectionError(f"Cannot connect to Ollama at {self.url}: {e}")
    
    def generate(self, prompt: str, temperature: float = 0.7, **kwargs) -> str:
        """Generate using Ollama"""
        try:
            response = self.session.post(
                f"{self.url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": temperature,
                    "stream": False
                },
                timeout=config.max_tool_timeout
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise


def get_llm_provider(provider: str = "google", **kwargs) -> LLMProvider:
    """Get LLM provider instance"""
    
    if provider == "google":
        return GoogleGeminiProvider(**kwargs)
    elif provider == "openai":
        return OpenAIProvider(**kwargs)
    elif provider == "ollama":
        return OllamaProvider(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def main():
    """Interactive LLM configuration"""
    import sys
    
    print("\n" + "="*70)
    print("FINANCIAL AGENT AI - LLM CONFIGURATION")
    print("="*70 + "\n")
    
    # Check current status
    status = LLMConfig.check_provider_status()
    
    print("CURRENT STATUS:")
    print("─"*70)
    for provider, info in status.items():
        icon = "✅" if info["configured"] else "❌"
        free = "💚 Free" if info["free_tier"] else "💳 Paid"
        print(f"{icon} {info['name']:<25} {free}")
    
    print("\n" + "="*70)
    print("AVAILABLE PROVIDERS:")
    print("="*70 + "\n")
    
    for i, (key, provider) in enumerate(LLMConfig.PROVIDERS.items(), 1):
        free = "💚 Free" if provider["free_tier"] else "💳 Paid"
        print(f"{i}. {provider['name']:<25} {free}")
    
    print("\nRECOMMENDATION:")
    print("─"*70)
    print("For development:  Use Google Gemini (free tier)")
    print("For production:   Use OpenAI GPT-4 (most reliable)")
    print("For local use:    Use Ollama (no API key needed)")
    
    # Interactive setup
    if len(sys.argv) > 1:
        provider_choice = sys.argv[1].lower()
        if provider_choice in LLMConfig.PROVIDERS:
            print("\n" + LLMConfig.get_setup_instructions(provider_choice))
            return
    
    print("\n" + "="*70)
    print("QUICK SETUP:")
    print("="*70 + "\n")
    
    print("Option 1: Google Gemini")
    print("  python config/llm_config.py google")
    print("\nOption 2: OpenAI GPT-4")
    print("  python config/llm_config.py openai")
    print("\nOption 3: Ollama")
    print("  python config/llm_config.py ollama")


if __name__ == "__main__":
    main()
