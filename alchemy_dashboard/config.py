# alchemy_dashboard/config.py
"""
Shared settings for the Alchemy Dashboard.

Other modules import values from here, e.g. `from .config import DB_NAME`.

Only DB_NAME is currently used by the app. The other settings below are
placeholders: nothing reads them yet, so changing them has no effect.
"""
import os

# === Application Configuration ===

# SQLite database filename.
# Full path to alchemy_experiments.db in the repository root (one folder above
# this file), so the same database is used no matter which folder the app is
# started from. db_utils.py defines the same path again; keep them in sync.
DB_NAME = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'alchemy_experiments.db'))

# Path settings (if needed later). Not currently used.
DATA_DIR = 'data/'
STATIC_DIR = 'static/'

# Feature toggles. Not currently used: the pages and routes do not check them,
# so downloads and time-series plots are always on.
ENABLE_EXPERIMENT_DOWNLOAD = True
ENABLE_TIME_SERIES_VIEW = True  # can turn off for MVP

# App metadata. Not currently used (page titles are set in the HTML templates).
APP_TITLE = "Alchemy Experiment Dashboard"
