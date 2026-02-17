# Lazy imports — avoid importing cassandra-driver at module load time.
# The driver requires asyncore (removed in Python 3.12) or libev C extension.
# Importing is deferred to when CassandraClient is actually used.

def get_cassandra():
    from .cassandra import get_cassandra as _get
    return _get()

def get_client_class():
    from .cassandra import CassandraClient
    return CassandraClient

__all__ = ['get_cassandra', 'get_client_class']