"""
Financial Agent AI - Root Entry Point
Thin delegation wrapper for professional project structure
"""

import asyncio
import sys
from core_main import main


if __name__ == "__main__":
    asyncio.run(main())
