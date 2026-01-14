from __future__ import annotations

import importlib
import sys


EXPECTED = {
    "numpy": "2.2.6",
    "pandas": "2.3.3",
    "pyarrow": "22.0.0",
    "xgboost": "1.7.6",
    "sklearn": "1.7.2",
    "matplotlib": "3.10.7",
    "seaborn": "0.13.2",
}


def _get_version(mod_name: str) -> str | None:
    if mod_name == "sklearn":
        import sklearn

        return getattr(sklearn, "__version__", None)
    mod = importlib.import_module(mod_name)
    return getattr(mod, "__version__", None)


def main() -> int:
    print("python", sys.version.replace("\n", " "))
    ok = True
    for mod, expected in EXPECTED.items():
        try:
            got = _get_version(mod)
        except Exception as e:
            print(mod, "MISSING", repr(e))
            ok = False
            continue
        status = "OK" if got == expected else "DIFF"
        if status != "OK":
            ok = False
        print(mod, got, f"(expected {expected})", status)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

