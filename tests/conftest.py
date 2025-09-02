import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(ROOT, ".."))
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)
