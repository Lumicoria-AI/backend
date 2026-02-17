import asyncio
import os
import sys
from typing import Dict, Any

# Add the project root to the python path
project_root = "/Users/ghgfd/Documents/LUMICORIA AI"
sys.path.append(project_root)

# Mock the logger to avoid clutter
import structlog
structlog.configure(
    processors=[structlog.processors.JSONRenderer()],
    logger_factory=structlog.PrintLoggerFactory(),
)

# Mock get_llm_client to avoid actual API calls
from unittest.mock import MagicMock
import backend.agents.base_agent
backend.agents.base_agent.get_llm_client = MagicMock()

# Import all agents
from backend.agents.research_mentor_agent import ResearchMentorAgent
from backend.agents.research_agent import ResearchAgent
from backend.agents.legal_document_agent import LegalDocumentAgent
from backend.agents.learning_coach_agent import LearningCoachAgent
from backend.agents.knowledge_graph_agent import KnowledgeGraphAgent
from backend.agents.ethics_bias_agent import EthicsBiasAgent
from backend.agents.focus_flow_agent import FocusFlowAgent
from backend.agents.workspace_ergonomics_agent import WorkspaceErgonomicsAgent
from backend.agents.customer_service_agent import CustomerServiceAgent
from backend.agents.data_analysis_agent import DataAnalysisAgent
from backend.agents.translation_agent import TranslationAgent
from backend.agents.meeting_fact_checker_agent import MeetingFactCheckerAgent

async def test_agent(agent_class, name):
    print(f"Testing {name}...")
    try:
        config = {
            "model_config": {"model": "gpt-4"},
            "system_prompt": "You are a helpful assistant."
        }
        agent = agent_class(config)
        print(f"  Instantiation: SUCCESS")
        
        # Test query_async
        # We expect a result or a handled error dictionary, but NOT an exception
        try:
            result = await agent.query_async("test query")
            print(f"  query_async: SUCCESS (Result: {list(result.keys())})")
        except NotImplementedError:
             print(f"  query_async: FAILED (NotImplementedError)")
             return False
        except Exception as e:
             print(f"  query_async: FAILED (Exception: {str(e)})")
             return False
             
        return True
    except Exception as e:
        print(f"  Instantiation: FAILED ({str(e)})")
        return False

async def main():
    agents = [
        (ResearchMentorAgent, "ResearchMentorAgent"),
        (ResearchAgent, "ResearchAgent"),
        (LegalDocumentAgent, "LegalDocumentAgent"),
        (LearningCoachAgent, "LearningCoachAgent"),
        (KnowledgeGraphAgent, "KnowledgeGraphAgent"),
        (EthicsBiasAgent, "EthicsBiasAgent"),
        (FocusFlowAgent, "FocusFlowAgent"),
        (WorkspaceErgonomicsAgent, "WorkspaceErgonomicsAgent"),
        (CustomerServiceAgent, "CustomerServiceAgent"),
        (DataAnalysisAgent, "DataAnalysisAgent"),
        (TranslationAgent, "TranslationAgent"),
        (MeetingFactCheckerAgent, "MeetingFactCheckerAgent"),
    ]
    
    results = []
    for agent_cls, name in agents:
        success = await test_agent(agent_cls, name)
        results.append((name, success))
    
    print("\n--- Summary ---")
    all_passed = True
    for name, success in results:
        status = "PASS" if success else "FAIL"
        if not success: all_passed = False
        print(f"{name}: {status}")
        
    if all_passed:
        print("\nAll agents verified successfully.")
    else:
        print("\nSome agents failed verification.")

if __name__ == "__main__":
    asyncio.run(main())
