"""Orracle Training — model training, export, comparison, and serving."""

from __future__ import annotations

import os

# Shared config directory (one level up from this package)
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')

from training import jobs
from training import log_parser
from training import hardware
from training import remote
from training import export_mgr
from training import generate
