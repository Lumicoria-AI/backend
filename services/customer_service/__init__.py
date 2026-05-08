"""Customer Service domain services.

Public functions in this package are the only entry points the API
endpoints should touch — endpoints stay thin, business rules live here.

Modules:
    tickets   : create / reply / status transitions
    templates : CRUD + 5-default seeder
    draft     : RAG-grounded AI draft context builder
    analytics : aggregation queries for /customer-service/analytics
    sanitize  : single source of truth for HTML sanitization
"""
