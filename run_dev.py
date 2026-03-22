#!/usr/bin/env python3
"""Orracle — dev server with auto-venv activation."""

import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
VENV_PYTHON = SCRIPT_DIR / 'venv' / 'bin' / 'python'

if VENV_PYTHON.exists() and sys.executable != str(VENV_PYTHON):
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON)] + sys.argv)

from app import app

if __name__ == '__main__':
    port = int(os.environ.get('FLASK_PORT', 5000))
    if '--port' in sys.argv:
        idx = sys.argv.index('--port')
        if idx + 1 < len(sys.argv):
            port = int(sys.argv[idx + 1])
    host = '0.0.0.0' if '--host' in sys.argv else '127.0.0.1'
    debug = os.environ.get('FLASK_DEBUG', '1') == '1'

    print(f"""
  orracle — AI Training Pipeline & Model Management
  http://{host}:{port}
  Debug: {debug}
""")

    app.run(host=host, port=port, debug=debug)
