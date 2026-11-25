import sys
from pathlib import Path

# Enable pytest processes to use packaging and release helpers.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
