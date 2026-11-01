from __future__ import annotations

from pathlib import Path
import importlib
import sys

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = {
    "numpy": None,
    "pandas": None,
    "sklearn": "scikit-learn",
    "scipy": None,
    "pyarrow": None,
    "duckdb": None,
    "openpyxl": None,
    "matplotlib": None,
}

def main():
    problems = []
    print("Runtime preflight")
    print("=================")
    print("Python:", sys.version.replace("\n", " "))

    for module, package in REQUIRED.items():
        try:
            m = importlib.import_module(module)
            ver = getattr(m, "__version__", "version not exposed")
            print(f"[PASS] {module}: {ver}")
        except Exception as e:
            problems.append(f"{module}: {e}")
            print(f"[FAIL] {module}: {e}")

    mimic_example = ROOT / "analysis" / "mimic" / "config.py.example"
    eicu_example = ROOT / "analysis" / "eicu" / "config.py.example"
    print("[PASS]" if mimic_example.exists() else "[FAIL]", "MIMIC config example")
    print("[PASS]" if eicu_example.exists() else "[FAIL]", "eICU config example")

    if problems:
        print("\nPRELIGHT FAILED")
        raise SystemExit(1)

    print("\nPRELIGHT PASS")
    print("Next: copy each config.py.example to config.py and set authorized local data paths.")

if __name__ == "__main__":
    main()
