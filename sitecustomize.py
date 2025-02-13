import sys
from pathlib import Path

# Add src to Python path for easier imports
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
