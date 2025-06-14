# sitecustomize.py （放在你运行 manage.py 的同一目录，或直接复制到 venv 的 site-packages）
print("🍀 sitecustomize loaded!", __file__)

# 1) 确保已经 pip install pysqlite3-binary
import pysqlite3

# 2) 把所有对 sqlite3 的引用都指向 pysqlite3
import sys
sys.modules['sqlite3'] = pysqlite3
sys.modules['sqlite3.dbapi2'] = pysqlite3