"""Shared pytest fixtures/configuration for the simulator's test suite.

The package code lives in ``src/clasher_new`` and opens its game-data JSON
files relative to the current working directory (see ``card_utils``), and its
modules import each other as top-level names (``from environment import ...``).
So every test runs with that directory on ``sys.path`` and as the CWD.
"""
import os
import sys

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CLASHER = os.path.join(_BASE, 'src', 'clasher_new')

sys.path.insert(0, _CLASHER)
# Done at import time (not in a fixture): pytest collects the test module —
# which imports ``environment``/``card_utils`` — before any fixture runs.
os.chdir(_CLASHER)

