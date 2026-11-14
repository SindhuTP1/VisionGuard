"""Isolate tests from real weights/runs. Settings are read at import time, so the
env vars must be set before anything imports ``app``."""
import os
import sys
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="visionguard-tests-"))
os.environ["VG_WEIGHTS_DIR"] = str(_TMP / "weights")
os.environ["VG_RUNS_DIR"] = str(_TMP / "runs")
os.environ["VG_DATA_DIR"] = str(_TMP / "data")
os.environ["VG_SKIP_WARMUP"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
