"""Smoke test — instantiate every agent class and verify no init crashes."""
import sys
from pathlib import Path

# Ensure the backend directory is on the path
backend_dir = Path(__file__).resolve().parent.parent
project_root = backend_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(backend_dir))

from backend.agents.agent_service import AgentService

# Minimal config — just type, no model (uses default provider)
ALL_AGENT_TYPES = [
    "document", "wellbeing", "meeting", "creative", "student",
    "rag", "meeting_fact_checker", "research", "research_mentor",
    "social_media", "legal_document", "learning_coach",
    "knowledge_graph", "ethics_bias", "focus_flow",
    "workspace_ergonomics", "customer_service", "data_analysis",
    "translation", "general",
]

config = {
    "agents": {
        name: {"type": name}
        for name in ALL_AGENT_TYPES
    }
}

try:
    service = AgentService(config)
except Exception as e:
    print(f"\n❌ AgentService init failed: {e}")
    sys.exit(1)

print(f"\n{'='*55}")
print(f"  Agent types registered: {len(service.agent_types)}")
print(f"  Agents loaded:          {len(service.agents)}")
print(f"{'='*55}")

for name, agent in sorted(service.agents.items()):
    print(f"  ✅ {name:25s} → {type(agent).__name__}")

failed = set(ALL_AGENT_TYPES) - set(service.agents.keys())
if failed:
    print(f"\n  ❌ Failed to load: {failed}")
    sys.exit(1)
else:
    print(f"\n🎉 All {len(service.agents)} agents initialized successfully!")
