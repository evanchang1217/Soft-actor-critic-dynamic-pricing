#!/usr/bin/env python

import sys

# ---- SQLITE MONKEY-PATCH 開始 ----
try:
    import pysqlite3
    # 用 pysqlite3 取代標準庫的 sqlite3
    sys.modules['sqlite3'] = pysqlite3
    sys.modules['sqlite3.dbapi2'] = pysqlite3
    print(f"🍀 sqlite3 已被 monkey-patch， 使用版本：{pysqlite3.sqlite_version}")
except ImportError:
    # 如果沒有安裝 pysqlite3-binary，就跳過
    pass



#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
