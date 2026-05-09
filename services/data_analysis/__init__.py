"""Data Analysis domain services.

Modules:
    runs       : CRUD over `data_analysis_runs`
    pipeline   : downloads from MinIO, parses with pandas, calls the agent
    analytics  : aggregation queries for /data-analysis/analytics
    nlq        : natural language Q&A on a run's stored data
    sanitize   : input sanitization helpers
"""
