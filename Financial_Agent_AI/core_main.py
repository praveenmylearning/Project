"""
Main entry point for Financial Agent AI
"""

import asyncio
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from agent.agent_loop import AgentLoop


async def main():
    """Main entry point"""
    logger.info("Financial Agent AI Starting...")
    
    # Initialize agent
    agent = AgentLoop(mode="hybrid", strategy="exploratory")
    
    # Example query
    query = "What is the intrinsic value of Apple stock using DCF analysis?"
    
    # Run agent
    result = await agent.run(query)
    
    # Print results
    print("\n" + "="*60)
    print("FINANCIAL AGENT AI - RESULTS")
    print("="*60)
    print(f"Query: {query}")
    print(f"Status: {result.get('status')}")
    print(f"Mode: {result.get('mode')}")
    
    if "phases" in result:
        phases = result["phases"]
        print(f"\nPerception Intent: {phases['perception'].get('data', {}).get('interpreted_intent')}")
        print(f"Decision Route: {phases['decision'].get('data', {}).get('reasoning')}")
        
        if phases['summarization']:
            summary = phases['summarization'].get('data', {}).get('summary', '')
            print(f"\nFinal Summary:\n{summary}")
    
    print("="*60)
    return result


if __name__ == "__main__":
    asyncio.run(main())
