"""Ethics & Bias Agent service layer.

Modules:
  - sanitize:    length caps + control-char strip for user content
  - repository:  Mongo-backed persistence of every analysis run
                 (multi-tenant on organization_id, soft-deleted)
  - orchestrator: wraps the agent call with permission scope, model
                  selection (Gemini / Claude / Perplexity / etc.),
                  and history persistence
"""
