"""Entry point kept for the Kaggle notebooks: run.py with the sibling-graph features on.

The graph stage lives in run.py behind `graph_enabled`, so the two drivers cannot drift.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run import main, parse_args  # noqa: E402

if __name__ == '__main__':
    main(parse_args(), overrides={'graph_enabled': True})
