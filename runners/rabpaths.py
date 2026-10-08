"""Shared locations for the RAB scripts. Everything is relative to the repository unless overridden.

  RAB_CASES  built cases (default <repo>/cases; created by rab/build_*.py)
  RAB_WORK   caches (embeddings, method features; default <repo>/work)
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES = os.environ.get('RAB_CASES', os.path.join(REPO, 'cases'))
MANIFEST = os.path.join(REPO, 'rab', 'manifest.json')
RESULTS = os.path.join(REPO, 'results')
WORK = os.environ.get('RAB_WORK', os.path.join(REPO, 'work'))
