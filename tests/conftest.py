import os
import sys

# Fix macOS fork() safety crash in datasets.map multiprocessing
if sys.platform == "darwin":
    os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")