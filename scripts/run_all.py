"""Điểm vào chạy các bước chính của dự án theo thứ tự."""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def run(module: str, *args: str) -> None:
    command = [sys.executable, "-m", module, *args]
    print("\n$", " ".join(command), flush=True)
    env = os.environ.copy()
    # Một số module trong src dùng import tương đối kiểu ``from common import``.
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    subprocess.run(command, cwd=ROOT, env=env, check=True)

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Chạy pipeline tái lập của dự án")
    parser.add_argument("--benchmark", action="store_true", help="chạy benchmark đầy đủ")
    args = parser.parse_args()
    run("src.data_quality_fairness")
    run("src.feature_selection_importance")
    if args.benchmark:
        run("src.run_unified_benchmark")
    print("\nHoàn tất pipeline.")

if __name__ == "__main__":
    main()
