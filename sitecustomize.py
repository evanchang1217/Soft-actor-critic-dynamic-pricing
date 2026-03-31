import sys

try:
    import pysqlite3
except ImportError:
    pysqlite3 = None

if pysqlite3 is not None:
    sys.modules["sqlite3"] = pysqlite3
    sys.modules["sqlite3.dbapi2"] = pysqlite3
