import os
import sys

# Add ./src to sys.path so `import review_assistant` works locally like in CI
ROOT = os.path.dirname(__file__)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)