"""Knowledge Graph domain services.

Modules:
    repository  : load + save per-org graphs (networkx DiGraph <-> SQL)
    extractions : audit log of every extract / discover / fill_gaps run
    analytics   : real stats backing /knowledge-graph/stats
    sanitize    : length caps + control-char stripping for user inputs
"""
