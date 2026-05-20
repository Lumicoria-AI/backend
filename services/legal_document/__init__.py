"""Legal Document Agent service layer.

Surface modules:
  - sanitize:    length caps + control-char strip for user-supplied text
  - repository:  Mongo-backed persistence of every analysis (multi-tenant)
  - orchestrator: wraps the agent call with permission scope, model
                  selection (Gemini / Claude / Perplexity), and history
                  persistence
"""
